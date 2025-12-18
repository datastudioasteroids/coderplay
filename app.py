# app.py
import os
import streamlit as st
from document_processor import extract_text_from_file
from ai_engine import get_llm_response

st.set_page_config(page_title="SaludInteractiva AI", page_icon="🏥", layout="wide")

ROLES = {
    "Payamédico": {
        "icon": "🤡",
        "desc": "Usa humor y ternura. Ideal para niños o para bajar la ansiedad.",
        "prompt": "Eres un Payamédico. Explica el contenido de forma divertida, con rimas, juegos de palabras y mucha empatía. Tu prioridad es la alegría del paciente."
    },
    "Enfermero": {
        "icon": "🩺",
        "desc": "Práctico, atento y enfocado en los cuidados diarios.",
        "prompt": "Eres un Enfermero con años de experiencia. Tu tono es sereno, práctico y te enfocas en explicar los pasos a seguir y los cuidados preventivos."
    },
    "Doctor": {
        "icon": "👨‍⚕️",
        "desc": "Profesional, técnico y analítico. Basado en evidencia.",
        "prompt": "Eres un Doctor Especialista. Tu tono es formal, clínico y preciso. Analiza los datos del documento con rigor científico y terminología médica adecuada."
    }
}

if "user_name" not in st.session_state:
    st.session_state.user_name = ""

with st.sidebar:
    st.header("👤 Perfil de Usuario")
    name_input = st.text_input("¿Cómo te llamas?", value=st.session_state.user_name)
    if name_input:
        st.session_state.user_name = name_input

    st.divider()
    
    st.header("🏥 Configuración de IA")
    provider = st.selectbox("Proveedor de IA", ["Hugging Face", "Gemini", "Hugging Chat (email:pass)"])
    
    # Intentamos leer los secrets de forma robusta
    api_key = None
    if provider == "Gemini":
        api_key = st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else os.getenv("GEMINI_API_KEY")
        if api_key:
            st.success("✅ Gemini configurado via Secrets")
    elif provider.startswith("Hugging Chat"):
        # si el usuario elige usar login email:pass
        api_key = st.secrets.get("HUGGING_CHAT_LOGIN") if hasattr(st, "secrets") else os.getenv("HUGGING_CHAT_LOGIN")
        if api_key:
            st.success("✅ Hugging Chat login configurado via Secrets")
        else:
            st.info("Si vas a usar Hugging Chat, en Secrets pon: HUGGING_CHAT_LOGIN = \"email:password\"")
    else:
        # Hugging Face
        # aceptamos HF_TOKEN o HUGGINGFACEHUB_API_TOKEN
        api_key = st.secrets.get("HF_TOKEN") if hasattr(st, "secrets") else os.getenv("HF_TOKEN")
        if not api_key:
            api_key = st.secrets.get("HUGGINGFACEHUB_API_TOKEN") if hasattr(st, "secrets") else os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if api_key:
            st.success("✅ Hugging Face token detectado en Secrets/env")
        else:
            st.info("Añadí HF_TOKEN = \"hf_xxx...\" en Streamlit Secrets")

    st.divider()
    st.subheader("🎭 Personalidad del Asistente")
    rol_nombre = st.radio("¿Quién te atiende hoy?", list(ROLES.keys()))
    rol_info = ROLES[rol_nombre]
    st.info(f"**{rol_nombre}**: {rol_info['desc']}")
    
    st.divider()
    uploaded_file = st.file_uploader("Subir informe médico", type=["pdf", "docx", "txt"])

saludo = f", {st.session_state.user_name}" if st.session_state.user_name else ""
st.title(f"{rol_info['icon']} Consulta con tu {rol_nombre}{saludo}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Haz tu pregunta médica aquí..."):
    if not api_key:
        st.error(f"⚠️ Las credenciales para {provider} no están configuradas en los Secrets de Streamlit.")
    elif not uploaded_file:
        st.warning("Primero debes subir un documento para analizar.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner(f"El {rol_nombre} está analizando la información para ti..."):
                try:
                    context = extract_text_from_file(uploaded_file)
                    # Llamada al helper que maneja HF / Gemini / otros
                    response = get_llm_response(
                        provider=provider,
                        api_key=api_key,
                        context=context,
                        user_query=prompt,
                        system_instruction=rol_info["prompt"],
                        model_id=os.getenv("HF_MODEL_ID")  # opcional: definir modelo por env
                    )
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Hubo un problema técnico: {e}")

