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

# Inicialización de estado de sesión para el nombre del usuario
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

# Barra Lateral
with st.sidebar:
    st.header("👤 Perfil de Usuario")
    name_input = st.text_input("¿Cómo te llamas?", value=st.session_state.user_name)
    if name_input:
        st.session_state.user_name = name_input

    st.divider()
    
    st.header("🏥 Configuración de IA")
    provider = st.selectbox("Proveedor de IA", ["Gemini", "Hugging Chat"])
    
    # Obtención de credenciales desde Streamlit Secrets
    try:
        if provider == "Gemini":
            api_key = st.secrets["GEMINI_API_KEY"]
            st.success("✅ Gemini configurado via Secrets")
        else:
            # Para Hugging Chat esperamos un secreto llamado HUGGING_CHAT_LOGIN con formato "email:pass"
            api_key = st.secrets["HUGGING_CHAT_LOGIN"]
            st.success("✅ Hugging Chat configurado via Secrets")
    except Exception:
        st.error(f"❌ No se encontraron secretos para {provider} en la configuración.")
        api_key = None
    
    st.divider()
    
    st.subheader("🎭 Personalidad del Asistente")
    rol_nombre = st.radio("¿Quién te atiende hoy?", list(ROLES.keys()))
    rol_info = ROLES[rol_nombre]
    st.info(f"**{rol_nombre}**: {rol_info['desc']}")
    
    st.divider()
    uploaded_file = st.file_uploader("Subir informe médico", type=["pdf", "docx", "txt"])

# Area Principal
saludo = f", {st.session_state.user_name}" if st.session_state.user_name else ""
st.title(f"{rol_info['icon']} Consulta con tu {rol_nombre}{saludo}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input de usuario
if prompt := st.chat_input("Haz tu pregunta médica aquí..."):
    if not api_key:
        st.error(f"⚠️ Las credenciales de {provider} no están configuradas en los Secretos de Streamlit.")
    elif not uploaded_file:
        st.warning("Primero debes subir un documento para analizar.")
    else:
        # Mostrar mensaje de usuario
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generar respuesta
        with st.chat_message("assistant"):
            with st.spinner(f"El {rol_nombre} está analizando la información para ti..."):
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
                    st.error(f"Hubo un problema técnico: {e}")
