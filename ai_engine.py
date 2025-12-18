import os
import json
import requests
from typing import Optional

HF_ROUTER_BASE = "https://router.huggingface.co/hf-inference/models"
HF_WHOAMI_URL = "https://huggingface.co/api/whoami-v2"

DEFAULT_MODEL = os.getenv("HF_MODEL_ID", "gpt2")


# =========================
# DEBUG HELPERS
# =========================

def hf_whoami(token: str) -> dict:
    """Verifica si el token HF es válido"""
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(HF_WHOAMI_URL, headers=headers, timeout=10)
    try:
        return {
            "status": r.status_code,
            "body": r.json()
        }
    except Exception:
        return {
            "status": r.status_code,
            "body": r.text
        }


def hf_router_test(token: str, model_id: str) -> dict:
    """Prueba conectividad mínima al HF Router"""
    url = f"{HF_ROUTER_BASE}/{model_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": "Test de conectividad",
        "parameters": {"max_new_tokens": 5},
        "options": {"wait_for_model": True}
    }
    r = requests.post(url, headers=headers, json=payload, timeout=20)
    try:
        return {
            "status": r.status_code,
            "body": r.json()
        }
    except Exception:
        return {
            "status": r.status_code,
            "body": r.text
        }


# =========================
# CORE LOGIC
# =========================

def build_prompt(system_instruction: str, context: str, user_query: str) -> str:
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

    # ===== 1. Token =====
    token = (
        api_key
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    if not token:
        return "❌ DEBUG: No se encontró HF_TOKEN en Secrets ni en variables de entorno."

    debug_info = {}

    # ===== 2. WHOAMI CHECK =====
    whoami = hf_whoami(token)
    debug_info["whoami"] = whoami

    if whoami["status"] != 200:
        return (
            "❌ DEBUG: El token de Hugging Face NO es válido.\n\n"
            f"WHOAMI STATUS: {whoami['status']}\n"
            f"WHOAMI BODY:\n{json.dumps(whoami['body'], indent=2, ensure_ascii=False)}"
        )

    # ===== 3. MODEL =====
    model = model_id or DEFAULT_MODEL
    debug_info["model"] = model

    # ===== 4. ROUTER TEST =====
    router_test = hf_router_test(token, model)
    debug_info["router_test"] = router_test

    if router_test["status"] != 200:
        return (
            "❌ DEBUG: Falló el acceso al HF Router.\n\n"
            f"MODEL: {model}\n"
            f"ROUTER STATUS: {router_test['status']}\n"
            f"ROUTER BODY:\n{json.dumps(router_test['body'], indent=2, ensure_ascii=False)}"
        )

    # ===== 5. REAL REQUEST =====
    prompt = build_prompt(system_instruction, context, user_query)

    url = f"{HF_ROUTER_BASE}/{model}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_tokens},
        "options": {"wait_for_model": True}
    }

    r = requests.post(url, headers=headers, json=payload, timeout=60)

    try:
        r.raise_for_status()
        response = r.json()
    except Exception as e:
        return (
            "❌ DEBUG: Error HTTP en inferencia.\n\n"
            f"STATUS: {r.status_code}\n"
            f"TEXT:\n{r.text}\n"
            f"EXCEPTION:\n{e}"
        )

    # ===== 6. NORMALIZE RESPONSE =====
    if isinstance(response, list) and response and "generated_text" in response[0]:
        return response[0]["generated_text"]

    if isinstance(response, dict):
        if "generated_text" in response:
            return response["generated_text"]
        if "error" in response:
            return f"❌ DEBUG: Error del modelo\n{response['error']}"

    return (
        "⚠️ DEBUG: Respuesta inesperada del modelo.\n\n"
        f"{json.dumps(response, indent=2, ensure_ascii=False)}"
    )

