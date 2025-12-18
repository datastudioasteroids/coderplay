
import streamlit as st
from document_processor import extract_text_from_file
from ai_engine import get_gemini_response

# Configuración de la página (estilo Streamlit)
st.set_page_config(page_title="DocuQuery AI", page_icon="📄", layout="wide")

# Estilo personalizado similar al front original
st.markdown("""
    <style>
    .stApp { background-color: white; }
    .main .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_password=True)

# Barra lateral (Sidebar)
with st.sidebar:
    st.title("⚙️ Configuración")
    api_key = st.text_input("Ingresa tu Gemini API Key", type="password")
    uploaded_file = st.file_uploader("Sube un documento", type=["pdf", "docx", "txt"])
    
    if uploaded_file:
        st.success(f"Archivo '{uploaded_file.name}' cargado.")

# Cuerpo principal
st.title("Pregunta a tus documentos 📄")
st.info("Sube un archivo en la barra lateral y comienza a chatear con su contenido.")

# Inicializar historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes del historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Lógica de chat
if prompt := st.chat_input("Escribe tu pregunta sobre el documento..."):
    if not api_key:
        st.error("Por favor, ingresa tu API Key en la barra lateral.")
    elif not uploaded_file:
        st.error("Por favor, sube un archivo primero.")
    else:
        # Agregar mensaje del usuario al chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Procesar respuesta
        with st.chat_message("assistant"):
            with st.spinner("Analizando documento..."):
                # 1. Extraer texto
                text_context = extract_text_from_file(uploaded_file)
                # 2. Obtener respuesta de IA
                response = get_gemini_response(api_key, text_context, prompt)
                st.markdown(response)
        
        # Guardar respuesta en el historial
        st.session_state.messages.append({"role": "assistant", "content": response})
