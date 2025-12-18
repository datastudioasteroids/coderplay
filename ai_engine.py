import google.generativeai as genai
import time

def get_gemini_response(api_key, context, user_query):
    """
    Configura Gemini y obtiene una respuesta basada en el contexto del documento.
    """
    genai.configure(api_key=api_key)
    
    # Configuración del modelo
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    system_prompt = f"""
    Eres un asistente experto en análisis de documentos. 
    Responde a la pregunta del usuario basándote estrictamente en el contenido proporcionado.
    Si la respuesta no está en el texto, indícalo educadamente.
    
    CONTENIDO DEL DOCUMENTO:
    {context}
    """
    
    # Intento de generación con lógica de reintentos básica
    try:
        response = model.generate_content([system_prompt, user_query])
        return response.text
    except Exception as e:
        return f"Error al procesar con la IA: {str(e)}"
