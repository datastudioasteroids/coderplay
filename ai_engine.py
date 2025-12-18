import google.generativeai as genai
from hugchat import hugchat
from hugchat.login import Login
import time

def get_llm_response(provider, api_key, context, user_query, system_instruction):
    """
    Obtiene respuesta del proveedor seleccionado (Gemini o Hugging Chat)
    aplicando el rol médico correspondiente.
    """
    
    full_prompt = f"{system_instruction}\n\nCONTEXTO DEL DOCUMENTO:\n{context}\n\nPREGUNTA DEL USUARIO: {user_query}"

    if provider == "Gemini":
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            return f"Error con Gemini: {str(e)}"

    elif provider == "Hugging Chat":
        try:
            # En Hugging Chat, el api_key suele ser el correo; se asume formato "email:password"
            if ":" not in api_key:
                return "Para Hugging Chat, usa el formato 'email:password' en el campo de clave."
            
            email, password = api_key.split(":")
            sign = Login(email, password)
            cookies = sign.login()
            
            chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
            # Creamos un nuevo flujo de conversación
            id = chatbot.new_conversation()
            chatbot.change_conversation(id)
            
            response = chatbot.chat(full_prompt)
            return str(response)
        except Exception as e:
            return f"Error con Hugging Chat: {str(e)}"
    
    return "Proveedor no soportado."
