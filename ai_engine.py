# ai_engine.py
import os
import json
from typing import Optional, Dict, Any, List

import requests

# Endpoints
HF_ROUTER_BASE = "https://router.huggingface.co/hf-inference/models"
HF_WHOAMI_URL = "https://huggingface.co/api/whoami-v2"
HF_MODELS_LIST_URL = "https://huggingface.co/api/models"

# No fallback a gpt2: leer HF_MODEL_ID de Secrets/ENV
DEFAULT_MODEL = os.getenv("HF_MODEL_ID")  # Debe estar configurada en Streamlit Secrets

# Intentar usar la SDK oficial si está instalada
try:
    from huggingface_hub import InferenceClient  # type: ignore
    HAS_INFERENCE_CLIENT = True
except Exception:
    HAS_INFERENCE_CLIENT = False


# -------------------------
# Utilidades de diagnóstico / helpers
# -------------------------
def _whoami(token: str) -> Dict[str, Any]:
    """Verifica validez del token (whoami)."""
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


def _list_models_for_provider(token: str, provider: str = "hf-inference", limit: int = 30) -> List[Dict[str, Any]]:
    """
    Lista modelos en el Hub filtrando por inference_provider.
    Devuelve una lista corta de candidatos (id, pipeline_tag).
    """
    params = {"inference_provider": provider, "limit": limit}
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = requests.get(HF_MODELS_LIST_URL, params=params, headers=headers, timeout=20)
        r.raise_for_status()
        models = r.json()
    except Exception as e:
        return [{"id": None, "error": f"No se pudo listar modelos: {e}"}]

    candidates = []
    for m in models:
        pipeline = m.get("pipeline_tag") or ""
        if pipeline in ("text-generation", "text-to-text-generation"):
            candidates.append({"id": m.get("id"), "pipeline_tag": pipeline})
        if len(candidates) >= 10:
            break

    if not candidates:
        # devolver algunos modelos por defecto (si no se encontraron text-generation)
        simple = []
        for m in models[:10]:
            simple.append({"id": m.get("id"), "pipeline_tag": m.get("pipeline_tag")})
        return simple

    return candidates


# -------------------------
# Construcción y llamadas de inferencia
# -------------------------
def _build_prompt(system_instruction: str, context: str, user_query: str) -> str:
    parts = []
    if system_instruction:
        parts.append(f"SISTEMA:\n{system_instruction}\n")
    if context:
        parts.append(f"DOCUMENTO:\n{context}\n")
    parts.append(f"PREGUNTA:\n{user_query}\n\nRESPUESTA:")
    return "\n".join(parts)


def _call_router_http(token: str, model: str, prompt: str, max_tokens: int = 300) -> Dict[str, Any]:
    """
    Llamada HTTP directa al HF Router.
    """
    url = f"{HF_ROUTER_BASE}/{model}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_tokens},
        "options": {"wait_for_model": True}
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    try:
        body = r.json()
    except Exception:
        body = r.text
    return {"ok": r.ok, "status": r.status_code, "body": body}


def _call_inference_client(token: str, model: str, prompt: str, max_tokens: int = 300) -> Any:
    """
    Llamada usando huggingface_hub.InferenceClient.
    Puede devolver diferentes shapes según la versión de la SDK/modelo.
    """
    client = InferenceClient(token=token)
    # Usamos text_generation; la firma puede variar según versión
    return client.text_generation(model=model, inputs=prompt, max_new_tokens=max_tokens)


def _extract_generated_text(resp: Any) -> Optional[str]:
    """
    Normaliza formas comunes de respuesta:
    - lista de dicts con 'generated_text'
    - dict con 'generated_text'
    - otros: devuelve string de JSON del objeto
    """
    try:
        if isinstance(resp, list) and resp and isinstance(resp[0], dict):
            # Ej: [{"generated_text": "..."}]
            if "generated_text" in resp[0]:
                return resp[0]["generated_text"]
        if isinstance(resp, dict):
            if "generated_text" in resp:
                return resp["generated_text"]
            # a veces la SDK devuelve {'generated_texts': [...]}
            if "generated_texts" in resp and isinstance(resp["generated_texts"], list) and resp["generated_texts"]:
                return resp["generated_texts"][0]
    except Exception:
        pass
    # fallback: stringificar
    try:
        return json.dumps(resp, ensure_ascii=False)
    except Exception:
        return str(resp)


# -------------------------
# Función principal expuesta
# -------------------------
def get_llm_response(
    provider: str,
    api_key: Optional[str],
    context: str,
    user_query: str,
    system_instruction: str = "",
    model_id: Optional[str] = None,
    max_tokens: int = 300
) -> str:
    """
    Obtiene la respuesta del LLM.
    - provider: solo informativo aquí (soporta 'Hugging Face' en esta implementación).
    - api_key: token (si se pasa desde el caller). Si es None, se busca HF_TOKEN en env.
    - model_id: obliga a usar ese modelo; si es None, usa DEFAULT_MODEL (de env).
    Devuelve un string listo para mostrarse en la UI (en caso de error devuelve mensaje explicativo).
    """

    # 1) token
    token = api_key or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        return "❌ ERROR: No se encontró HF_TOKEN. Agregá HF_TOKEN en Streamlit Secrets."

    # 2) modelo: obligatorio (no fallback a gpt2)
    model = model_id or DEFAULT_MODEL
    if not model:
        return (
            "❌ ERROR: No se encontró HF_MODEL_ID en Streamlit Secrets ni fue pasado como model_id.\n"
            "Indica HF_MODEL_ID en Streamlit Secrets con un modelo expuesto por Inference Providers."
        )

    # 3) whoami (valida token)
    who = _whoami(token)
    if not who.get("ok"):
        body = who.get("body")
        return (
            "❌ ERROR: Token inválido o whoami falló.\n"
            f"WHOAMI status: {who.get('status')}\nWHOAMI body: {json.dumps(body, indent=2, ensure_ascii=False)}\n\n"
            "Acción: regenerá un token en https://huggingface.co/settings/tokens y actualizá HF_TOKEN en Secrets."
        )

    # 4) quick router test
    router_test = _call_router_http(token, model, "Connectivity test", max_tokens=4)
    if not router_test.get("ok"):
        status = router_test.get("status")
        body = router_test.get("body")
        # si 404 -> listar candidatos útiles y sugerir cambio de modelo
        if status == 404:
            candidates = _list_models_for_provider(token, provider="hf-inference", limit=30)
            # transformar candidates a texto
            if candidates and candidates[0].get("error"):
                candidate_text = f"No se pudieron obtener candidatos: {candidates[0].get('error')}"
            else:
                candidate_text = "\n".join([f"- {c.get('id')} (pipeline={c.get('pipeline_tag')})" for c in candidates])
            return (
                "❌ ERROR: HF Router devolvió 404 para el modelo solicitado.\n\n"
                f"MODEL: {model}\nROUTER STATUS: {status}\nROUTER BODY: {json.dumps(body, ensure_ascii=False)}\n\n"
                "Modelos públicos disponibles vía Inference Providers (ejemplos):\n"
                f"{candidate_text}\n\n"
                "Acción: reemplazá HF_MODEL_ID en Streamlit Secrets por uno de los modelos listados arriba y redeploy."
            )
        # otros errores
        return (
            "❌ ERROR: Falló la conexión inicial al HF Router.\n\n"
            f"STATUS: {status}\nBODY: {json.dumps(body, ensure_ascii=False)}\n\n"
            "Acción: revisá el token, permisos y la conectividad."
        )

    # 5) construir prompt y llamar (preferir SDK si está disponible)
    prompt = _build_prompt(system_instruction, context, user_query)

    # Intentar SDK InferenceClient primero si está instalado
    if HAS_INFERENCE_CLIENT:
        try:
            sdk_resp = _call_inference_client(token, model, prompt, max_tokens=max_tokens)
            text = _extract_generated_text(sdk_resp)
            if text:
                return text
            # si SDK devolvió algo inesperado, intentar convertir a str/JSON
            return json.dumps(sdk_resp, ensure_ascii=False)
        except Exception:
            # si falla SDK, caemos al router HTTP
            pass

    # Fallback: llamada HTTP al router para inferencia
    final = _call_router_http(token, model, prompt, max_tokens=max_tokens)
    if not final.get("ok"):
        return (
            "❌ ERROR: Error HTTP en la inferencia final.\n\n"
            f"STATUS: {final.get('status')}\nBODY: {json.dumps(final.get('body'), ensure_ascii=False)}\n\n"
            "Acción: revisá que HF_MODEL_ID esté habilitado para Inference Providers o regenerá el token."
        )

    # Normalizar respuesta
    body = final.get("body")
    text = _extract_generated_text(body)
    if text:
        return text
    return f"⚠️ Respuesta inesperada del modelo: {json.dumps(body, indent=2, ensure_ascii=False)}"


