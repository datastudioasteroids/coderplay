# ai_engine.py
import os
import json
import requests
from typing import Optional

DEFAULT_MODEL = os.getenv("HF_MODEL_ID", "gpt2")

HF_ROUTER_URL = "https://router.huggingface.co/hf-inference/models"

def hf_inference_requests(model_id: str, token: str, inputs: str, max_tokens: int = 200):
    url = f"{HF_ROUTER_URL}/{model_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": inputs,
        "parameters": {"max_new_tokens": max_tokens},
        "options": {"wait_for_model": True}
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()

def format_prompt(system_instruction: str, context: str, user_query: str) -> str:
    parts = []
    if system_instruction:
        parts.append(f"SISTEMA:\n{system_instruction}\n")
    if context:
        parts.append(f"DOCUMENTO:\n{context}\n")
    parts.append(f"PREGUNTA:\n{user_query}\n\nRESPUESTA:")
    return "\n".join(parts)

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
        raise ValueError("No se encontró HF_TOKEN en secrets ni env.")

    model = model_id or DEFAULT_MODEL
    prompt = format_prompt(system_instruction, context, user_query)

    try:
        resp = hf_inference_requests(model, token, prompt, max_tokens)

        # Respuesta típica: [{"generated_text": "..."}]
        if isinstance(resp, list) and resp and "generated_text" in resp[0]:
            return resp[0]["generated_text"]

        # Otros formatos posibles
        if isinstance(resp, dict):
            if "generated_text" in resp:
                return resp["generated_text"]
            if resp.get("error"):
                raise RuntimeError(resp["error"])

        return json.dumps(resp, ensure_ascii=False)

    except requests.HTTPError as he:
        detail = he.response.text if he.response is not None else ""
        raise RuntimeError(f"HF Router HTTP Error: {he} - {detail}")

