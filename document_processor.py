import docx2txt
import PyPDF2
import io

def extract_text_from_file(uploaded_file):
    """
    Extrae el texto de archivos cargados según su extensión.
    Soporta: .txt, .pdf, .docx
    """
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension == 'txt':
        return str(uploaded_file.read(), "utf-8")
        
    elif file_extension == 'pdf':
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
        
    elif file_extension == 'docx':
        return docx2txt.process(uploaded_file)
        
    else:
        return None
