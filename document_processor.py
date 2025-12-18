# document_processor.py
from io import BytesIO
import PyPDF2
import docx

def extract_text_from_file(uploaded_file) -> str:
    """
    uploaded_file: objeto tipo UploadedFile de Streamlit
    Devuelve texto concatenado del archivo (txt, pdf, docx).
    """
    if not uploaded_file:
        return ""
    filename = uploaded_file.name.lower()
    content = uploaded_file.read()

    if filename.endswith(".txt"):
        try:
            return content.decode("utf-8", errors="replace")
        except Exception:
            return content.decode("latin-1", errors="replace")

    if filename.endswith(".pdf"):
        text_parts = []
        try:
            reader = PyPDF2.PdfReader(BytesIO(content))
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
            return "\n".join(text_parts)
        except Exception as e:
            # fallback simple
            return f"[No se pudo extraer PDF: {e}]"

    if filename.endswith(".docx"):
        try:
            doc = docx.Document(BytesIO(content))
            paragraphs = [p.text for p in doc.paragraphs if p.text]
            return "\n".join(paragraphs)
        except Exception as e:
            return f"[No se pudo extraer DOCX: {e}]"

    # fallback: intentar decode
    try:
        return content.decode("utf-8", errors="replace")
    except Exception:
        return str(content)

