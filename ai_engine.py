# ai_engine.py
import os
import json
import requests
from typing import Optional

DEFAULT_MODEL = os.getenv("HF_MODEL_ID", "gpt2")

def hf_inference_requests(model_id: str, token: str, inputs: str, max_tokens: int = 200):
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": inputs,
        "parameters": {"max_new_tokens": max_tokens},
        "options": {"wait_for_model": True}
    }
    resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()

def format_prompt(system_instruction: str, context: str, user_query: str) -> str:
    parts = []
    if system_instruction:
        parts.append(f"SISTEMA: {system_instruction}\n\n")
    if context:
        parts.append(f"DOCUMENTO:\n{context}\n\n")
    parts.append(f"PREGUNTA:\n{user_query}\n\nRESPUESTA:")
    return "\n".join(parts)

def get_llm_response(provider: str,
                     api_key: Optional[str],
                     context: str,
                     user_query: str,
                     system_instruction: str = "",
                     model_id: Optional[str] = None,
                     max_tokens: int = 300) -> str:
    # Forzamos Hugging Face cuando se provee HF token o provider indica HF
    token = api_key or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise ValueError("No se encontró HF token (HF_TOKEN o HUGGINGFACEHUB_API_TOKEN).")
    model = model_id or DEFAULT_MODEL
    prompt = format_prompt(system_instruction, context, user_query)
    try:
        resp = hf_inference_requests(model, token, prompt, max_tokens=max_tokens)
        # Normalizar respuesta típica
        if isinstance(resp, list) and resp and isinstance(resp[0], dict):
            if "generated_text" in resp[0]:
                return resp[0]["generated_text"]
            return json.dumps(resp, ensure_ascii=False)
        if isinstance(resp, dict):
            # algunos modelos devuelven {'generated_text': '...'}
            if "generated_text" in resp:
                return resp["generated_text"]
            if resp.get("error"):
                raise RuntimeError(resp["error"])
            return json.dumps(resp, ensure_ascii=False)
        return str(resp)
    except requests.HTTPError as he:
        # incluir contenido de respuesta si está
        content = getattr(he.response, "text", "")
        raise RuntimeError(f"Error HTTP al llamar HF Inference API: {he} - {content}")
    except Exception as e:
        raise RuntimeError(e)

