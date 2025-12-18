import streamlit as st
from document_processor import extract_text_from_file
from ai_engine import get_llm_response

# Configuración de la página
st.set_page_config(page_title="SaludInteractiva AI", page_icon="🏥", layout="wide")

# Definición de personalidades (Carácter de los roles)
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

# Barra Lateral
with st.sidebar:
    st.header("🏥 Configuración")
    provider = st.selectbox("Elegir IA", ["Gemini", "Hugging Chat"])
    
    help_text = "API Key para Gemini" if provider == "Gemini" else "Formato 'email:password'"
    api_key = st.text_input(f"Credenciales ({provider})", type="password", help=help_text)
    
    st.divider()
    
    st.subheader("🎭 Personalidad del Asistente")
    rol_nombre = st.radio("¿Quién te atiende hoy?", list(ROLES.keys()))
    rol_info = ROLES[rol_nombre]
    st.info(f"**{rol_nombre}**: {rol_info['desc']}")
    
    st.divider()
    uploaded_file = st.file_uploader("Subir informe médico", type=["pdf", "docx", "txt"])

# Area Principal
st.title(f"{rol_info['icon']} Consulta con tu {rol_nombre}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input de usuario
if prompt := st.chat_input("Haz tu pregunta médica aquí..."):
    if not api_key:
        st.warning(f"Faltan las credenciales de {provider}")
    elif not uploaded_file:
        st.warning("Primero debes subir un documento para analizar.")
    else:
        # Mostrar mensaje de usuario
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generar respuesta
        with st.chat_message("assistant"):
            with st.spinner(f"El {rol_nombre} está leyendo el documento..."):
                try:
                    context = extract_text_from_file(uploaded_file)
                    response = get_llm_response(
                        provider=provider,
                        api_key=api_key,
                        context=context,
                        user_query=prompt,
                        system_instruction=rol_info["prompt"]
                    )
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Hubo un problema: {e}")
