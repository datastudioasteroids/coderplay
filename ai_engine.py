# ai_engine.py
import os
import json
import requests
from typing import Optional, List, Dict

HF_ROUTER_BASE = "https://router.huggingface.co/hf-inference/models"
HF_WHOAMI_URL = "https://huggingface.co/api/whoami-v2"
HF_MODELS_LIST_URL = "https://huggingface.co/api/models"  # use query param inference_provider

DEFAULT_MODEL = os.getenv("HF_MODEL_ID", "gpt2")


# -----------------------
# Helpers: whoami + router test + list models
# -----------------------
def hf_whoami(token: str) -> Dict:
    headers = {"Authorization": f"Bearer {token}"}
    try:
        r = requests.get(HF_WHOAMI_URL, headers=headers, timeout=10)
        try:
            body = r.json()
        except Exception:
            body = r.text
        return {"ok": r.ok, "status": r.status_code, "body": body}
    except Exception as e:
        return {"ok": False, "status": None, "body": str(e)}


def hf_router_test(token: str, model_id: str) -> Dict:
    url = f"{HF_ROUTER_BASE}/{model_id}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "inputs": "Test de conectividad",
        "parameters": {"max_new_tokens": 5},
        "options": {"wait_for_model": True}
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=20)
        try:
            body = r.json()
        except Exception:
            body = r.text
        return {"ok": r.ok, "status": r.status_code, "body": body}
    except requests.RequestException as e:
        return {"ok": False, "status": None, "body": str(e)}


def list_inference_models(token: str, provider: str = "hf-inference", limit: int = 30) -> List[Dict]:
    """
    Consulta el Hub para listar modelos con `inference_provider=<provider>`.
    Filtra y devuelve una lista corta de modelos con pipeline_tag 'text-generation'
    o similares.
    """
    params = {
        "inference_provider": provider,
        "limit": limit,
        "sort": "downloads",
        "direction": -1
    }
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = requests.get(HF_MODELS_LIST_URL, params=params, headers=headers, timeout=20)
        r.raise_for_status()
        models = r.json()  # lista de objetos con campo "id", "pipeline_tag", "tags", etc.
    except Exception as e:
        # fallback: devolver mensaje en lista para mostrar al usuario
        return [{"id": None, "error": f"No se pudo listar modelos: {e}"}]

    # Filtrar modelos útiles para text generation
    candidates = []
    for m in models:
        pipeline = m.get("pipeline_tag") or ""
        tags = m.get("tags") or []
        if pipeline in ("text-generation", "text-to-text-generation", "text-generation-inference") or \
           "text-generation" in tags or "text-to-text-generation" in tags:
            candidates.append({"id": m.get("id"), "pipeline_tag": pipeline, "tags": tags})
        # stop early if enough candidates
        if len(candidates) >= 10:
            break

    # Si no encontramos candidatos text-generation, devolver algunos repos por downloads
    if not candidates:
        # devolver los primeros 10 ids simples
        simple = []
        for m in models[:10]:
            simple.append({"id": m.get("id"), "pipeline_tag": m.get("pipeline_tag"), "tags": m.get("tags")})
        return simple

    return candidates


# -----------------------
# Prompt builder + inference call
# -----------------------
def build_prompt(system_instruction: str, context: str, user_query: str) -> str:
    parts = []
    if system_instruction:
        parts.append(f"SISTEMA:\n{system_instruction}\n")
    if context:
        parts.append(f"DOCUMENTO:\n{context}\n")
    parts.append(f"PREGUNTA:\n{user_query}\n\nRESPUESTA:")
    return "\n".join(parts)


def call_hf_router(token: str, model: str, prompt: str, max_tokens: int = 300) -> Dict:
    url = f"{HF_ROUTER_BASE}/{model}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens}, "options": {"wait_for_model": True}}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    # r.raise_for_status()  # manejamos estado abajo
    try:
        body = r.json()
    except Exception:
        body = r.text
    return {"ok": r.ok, "status": r.status_code, "body": body}


# -----------------------
# Función principal expuesta: get_llm_response
# -----------------------
def get_llm_response(
    provider: str,
    api_key: Optional[str],
    context: str,
    user_query: str,
    system_instruction: str = "",
    model_id: Optional[str] = None,
    max_tokens: int = 300
) -> str:

    token = api_key or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        return "❌ DEBUG: No se encontró HF_TOKEN en Secrets ni como variable de entorno."

    # 1) WHOAMI
    whoami = hf_whoami(token)
    if not whoami.get("ok"):
        return (
            "❌ DEBUG: WHOAMI falló. El token NO es válido.\n\n"
            f"WHOAMI STATUS: {whoami.get('status')}\nWHOAMI BODY:\n{json.dumps(whoami.get('body'), indent=2, ensure_ascii=False)}\n\n"
            "Acción recomendada: regenerá un token en https://huggingface.co/settings/tokens y pegalo en Streamlit Secrets como HF_TOKEN."
        )

    model = model_id or DEFAULT_MODEL

    # 2) Router quick test
    router = hf_router_test(token, model)
    if not router.get("ok"):
        # Si obtenemos 404 o similar: listar modelos disponibles y sugerir cambios
        if router.get("status") == 404:
            candidates = list_inference_models(token, provider="hf-inference")
            # Si la lista devolvió error
            if candidates and candidates[0].get("error"):
                return (
                    "❌ DEBUG: HF Router devolvió 404 para el modelo solicitado.\n\n"
                    f"MODEL: {model}\nROUTER STATUS: {router.get('status')}\nROUTER BODY: {json.dumps(router.get('body'), ensure_ascii=False)}\n\n"
                    "Además, falló la consulta para listar modelos disponibles:\n"
                    f"{candidates[0].get('error')}\n\n"
                    "Acción recomendada: 1) revisá que tu token sea válido (WHOAMI); 2) cambiá HF_MODEL_ID en Secrets por uno de los modelos disponibles para Inference Providers."
                )
            # formar mensaje con candidatos
            list_text = "\n".join([f"- {c.get('id')} (pipeline={c.get('pipeline_tag')})" for c in candidates])
            return (
                "❌ DEBUG: HF Router devolvió 404 para el modelo solicitado.\n\n"
                f"MODEL: {model}\nROUTER STATUS: {router.get('status')}\nROUTER BODY: {json.dumps(router.get('body'), ensure_ascii=False)}\n\n"
                "Modelos públicos disponibles vía Inference Providers (ejemplos):\n"
                f"{list_text}\n\n"
                "Acción recomendada: reemplazá HF_MODEL_ID en Streamlit Secrets por uno de los modelos listados arriba y redeploy.\n"
                "Ej: HF_MODEL_ID = \"HuggingFaceTB/SmolLM3-3B\""
            )
        # otros errores (timeout, 5xx)
        return (
            "❌ DEBUG: Error al conectar con HF Router.\n\n"
            f"STATUS: {router.get('status')}\nBODY: {json.dumps(router.get('body'), ensure_ascii=False)}\n\n"
            "Acción: revisá conectividad, cuotas y permisos de tu cuenta."
        )

    # 3) Si el router quick test OK -> hacemos la llamada real con el prompt
    prompt = build_prompt(system_instruction, context, user_query)
    result = call_hf_router(token, model, prompt, max_tokens=max_tokens)

    if not result.get("ok"):
        return (
            "❌ DEBUG: Error HTTP en inferencia final.\n\n"
            f"STATUS: {result.get('status')}\n"
            f"BODY: {json.dumps(result.get('body'), ensure_ascii=False)}\n\n"
            "Si el cuerpo muestra 'error' con texto 'Invalid username or password', regenerá tu token en https://huggingface.co/settings/tokens.\n"
            "Si muestra 'Not Found', probá cambiar HF_MODEL_ID por uno de los modelos disponibles para Inference Providers."
        )

    # Normalizar respuesta típica
    resp = result.get("body")
    if isinstance(resp, list) and resp and isinstance(resp[0], dict) and "generated_text" in resp[0]:
        return resp[0]["generated_text"]

    if isinstance(resp, dict):
        if "generated_text" in resp:
            return resp["generated_text"]
        if "error" in resp:
            return f"❌ DEBUG: Error del modelo: {resp['error']}"

    return f"⚠️ DEBUG: Respuesta inesperada:\n{json.dumps(resp, indent=2, ensure_ascii=False)}"

