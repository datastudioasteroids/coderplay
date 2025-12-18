# ai_engine.py
import os, json, requests
from typing import Optional

# Hugging Face endpoints
HF_ROUTER_BASE = "https://router.huggingface.co/hf-inference/models"
HF_WHOAMI_URL = "https://huggingface.co/api/whoami-v2"
HF_MODELS_LIST_URL = "https://huggingface.co/api/models"

DEFAULT_MODEL = os.getenv("HF_MODEL_ID", "HuggingFaceH4/zephyr-7b-beta")

# Try to import the official SDK
try:
    from huggingface_hub import InferenceClient
    HAS_INFERENCE_CLIENT = True
except Exception:
    HAS_INFERENCE_CLIENT = False


def whoami(token: str):
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(HF_WHOAMI_URL, headers=headers, timeout=10)
    try:
        return {"ok": r.ok, "status": r.status_code, "body": r.json()}
    except Exception:
        return {"ok": r.ok, "status": r.status_code, "body": r.text}


def router_health_check(token: str, model: str):
    url = f"{HF_ROUTER_BASE}/{model}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"inputs": "Connectivity test", "parameters": {"max_new_tokens": 4}, "options": {"wait_for_model": True}}
    r = requests.post(url, headers=headers, json=payload, timeout=20)
    try:
        return {"ok": r.ok, "status": r.status_code, "body": r.json()}
    except Exception:
        return {"ok": r.ok, "status": r.status_code, "body": r.text}


def list_models_for_provider(token: str, provider: str = "hf-inference", limit: int = 30):
    params = {"inference_provider": provider, "limit": limit}
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.get(HF_MODELS_LIST_URL, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    models = r.json()
    # Filter text-generation candidates
    candidates = []
    for m in models:
        if m.get("pipeline_tag") in ("text-generation", "text-to-text-generation"):
            candidates.append({"id": m.get("id"), "pipeline": m.get("pipeline_tag")})
        if len(candidates) >= 10:
            break
    return candidates if candidates else [{"id": m.get("id"), "pipeline": m.get("pipeline_tag")} for m in models[:10]]


def build_prompt(system_instruction: str, context: str, user_query: str) -> str:
    parts = []
    if system_instruction:
        parts.append(f"SISTEMA:\n{system_instruction}\n")
    if context:
        parts.append(f"DOCUMENTO:\n{context}\n")
    parts.append(f"PREGUNTA:\n{user_query}\n\nRESPUESTA:")
    return "\n".join(parts)


def call_router(token: str, model: str, prompt: str, max_tokens: int = 300):
    url = f"{HF_ROUTER_BASE}/{model}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens}, "options": {"wait_for_model": True}}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    try:
        return {"ok": r.ok, "status": r.status_code, "body": r.json()}
    except Exception:
        return {"ok": r.ok, "status": r.status_code, "body": r.text}


def call_with_inference_client(token: str, model: str, prompt: str, max_tokens: int = 300):
    client = InferenceClient(token=token)
    # text_generation via SDK - the SDK may return different shapes depending on version
    out = client.text_generation(model=model, inputs=prompt, max_new_tokens=max_tokens)
    return out


def get_llm_response(provider: str,
                     api_key: Optional[str],
                     context: str,
                     user_query: str,
                     system_instruction: str = "",
                     model_id: Optional[str] = None,
                     max_tokens: int = 400) -> str:
    token = api_key or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        return "❌ No se encontró HF_TOKEN en Secrets ni en variables de entorno. Pegá HF_TOKEN en Streamlit Secrets."

    # WHOAMI
    who = whoami(token)
    if not who.get("ok"):
        return f"❌ WHOAMI falló (token inválido). status={who.get('status')} body={who.get('body')}"

    model = model_id or DEFAULT_MODEL

    # Router quick test
    rtest = router_health_check(token, model)
    if not rtest.get("ok"):
        # if 404, list candidates and return helpful message
        if rtest.get("status") == 404:
            try:
                candidates = list_models_for_provider(token, provider="hf-inference", limit=30)
                cand_text = "\n".join([f"- {c['id']} (pipeline={c.get('pipeline')})" for c in candidates])
            except Exception as e:
                cand_text = f"No se pudo listar candidatos: {e}"
            return (f"❌ HF Router devolvió 404 para el modelo solicitado ({model}).\n"
                    f"ROUTER BODY: {rtest.get('body')}\n\nModelos disponibles (ejemplos):\n{cand_text}\n\n"
                    "Acción: reemplazar HF_MODEL_ID en Secrets por uno de los modelos listados.")
        return f"❌ Error al probar el Router: status={rtest.get('status')} body={rtest.get('body')}"

    # If InferenceClient is available use it
    prompt = build_prompt(system_instruction, context, user_query)
    if HAS_INFERENCE_CLIENT:
        try:
            out = call_with_inference_client(token, model, prompt, max_tokens=max_tokens)
            # Normalize common shapes
            if isinstance(out, dict) and out.get("generated_text"):
                return out["generated_text"]
            try:
                return str(out)
            except:
                return json.dumps(out, ensure_ascii=False)
        except Exception as e:
            # fallback to direct router call
            pass

    # fallback to direct router HTTP call
    res = call_router(token, model, prompt, max_tokens=max_tokens)
    if not res.get("ok"):
        return f"❌ Error HTTP inferencia final: status={res.get('status')} body={res.get('body')}"
    body = res.get("body")
    if isinstance(body, list) and body and isinstance(body[0], dict) and body[0].get("generated_text"):
        return body[0]["generated_text"]
    if isinstance(body, dict) and body.get("generated_text"):
        return body.get("generated_text")
    return f"⚠️ Respuesta inesperada del modelo: {json.dumps(body, indent=2, ensure_ascii=False)}"

