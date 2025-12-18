# ai_engine.py
import os
import json
from typing import Optional

# Intentamos usar huggingface_hub si está instalado
try:
    from huggingface_hub import InferenceClient
    HAS_HF_HUB = True
except Exception:
    HAS_HF_HUB = False

import requests

DEFAULT_MODEL = "gpt2"  # cambiar por el modelo que quieras o poner en secrets

def hf_call_inference_client(model_id: str, token: str, inputs: str, max_tokens: int = 200):
    client = InferenceClient(token=token)
    # text generation
    return client.text_generation(inputs, model=model_id, max_new_tokens=max_tokens)

def hf_call_requests(model_id: str, token: str, inputs: str, max_tokens: int = 200):
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
    # Construye prompt simple con instrucción de sistema + contexto + pregunta del usuario
    pieces = []
    if system_instruction:
        pieces.append(f"SISTEMA: {system_instruction}\n\n")
    if context:
        # acotar contexto si está muy largo sería ideal; acá lo pegamos completo
        pieces.append(f"DOCUMENTO:\n{context}\n\n")
    pieces.append(f"PREGUNTA:\n{user_query}\n\nRESPUESTA:")
    return "\n".join(pieces)

def get_llm_response(provider: str,
                     api_key: Optional[str],
                     context: str,
                     user_query: str,
                     system_instruction: str = "",
                     model_id: Optional[str] = None,
                     max_tokens: int = 300) -> str:
    """
    provider: "Gemini", "Hugging Face", "Hugging Chat" (tu elección)
    api_key: token o login (según provider)
    """
    if provider.lower().startswith("gemini"):
        # Placeholder: si tens integración con Gemini ponela aquí.
        raise NotImplementedError("Integración con Gemini no implementada en este helper.")
    
    # Si pedís Hugging Chat con login (email:pass) — esto usaría librería no oficial.
    if provider.lower().startswith("hugging chat") or provider.lower().startswith("huggingchat"):
        # Si tenés la librería hugchat y querés usarla: el secreto debe ser "email:pass"
        # Implementar aquí si realmente querés usar hugchat. Por seguridad, recomendamos usar HF_TOKEN.
        raise NotImplementedError("HuggingChat via email/password no implementado. Usá HF_TOKEN con Hugging Face Inference API.")
    
    # Default -> Hugging Face Inference API usando HF_TOKEN
    token = api_key or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise ValueError("No se encontró HF token (HF_TOKEN o HUGGINGFACEHUB_API_TOKEN).")
    
    model = model_id or os.getenv("HF_MODEL_ID") or DEFAULT_MODEL
    prompt = format_prompt(system_instruction, context, user_query)

    # Preferir huggingface_hub.InferenceClient si está disponible
    try:
        if HAS_HF_HUB:
            resp = hf_call_inference_client(model, token, prompt, max_tokens=max_tokens)
            # InferenceClient.text_generation devuelve dict/obj dependiendo de versión
            # intentamos normalizar:
            if isinstance(resp, dict):
                # algunas versiones devuelven {'generated_text': '...'}
                if "generated_text" in resp:
                    return resp["generated_text"]
                # o {"generated_texts": [...]}
                if "generated_texts" in resp and resp["generated_texts"]:
                    return resp["generated_texts"][0]
                # o dict con 'results'
            # otras versiones devuelven un objeto complejo; convertir a string si es necesario
            try:
                return str(resp)
            except Exception:
                return json.dumps(resp)
        else:
            resp = hf_call_requests(model, token, prompt, max_tokens=max_tokens)
            # respuestas comunes: list with dict containing 'generated_text'
            if isinstance(resp, list) and resp and isinstance(resp[0], dict):
                if "generated_text" in resp[0]:
                    return resp[0]["generated_text"]
                # si la estructura es diferente, intentar stringify
                return json.dumps(resp)
            if isinstance(resp, dict) and resp.get("error"):
                raise RuntimeError(resp["error"])
            return json.dumps(resp)
    except requests.HTTPError as he:
        raise RuntimeError(f"Error HTTP al llamar HF Inference API: {he} - {getattr(he, 'response', None)}")
    except Exception as e:
        raise RuntimeError(f"Error al obtener respuesta del modelo: {e}")
