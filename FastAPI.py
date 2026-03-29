from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import ollama
import docx  # python-docx
import PyPDF2
from odf import text, load
from io import BytesIO
import os

def decode_file_contents(contents: bytes) -> str:
    encodings_to_try = [
        'utf-8',
        'cp1251',  # Windows-1251
        'koi8-r',  # Русская KOI8
        'iso-8859-5',  # Latin/Cyrillic
        'windows-1252',  # Западная Европа
        'ascii',  # Только ASCII
    ]

    for encoding in encodings_to_try:
        try:
            return contents.decode(encoding)
        except UnicodeDecodeError:
            continue

    # Если ни одна кодировка не подошла, ищем читаемые фрагменты
    try:
        # Пытаемся декодировать как UTF-8, заменяя проблемные символы
        return contents.decode('utf-8', errors='replace')
    except:
        raise HTTPException(
            status_code=400,
            detail="Unable to decode file: unsupported encoding. Please provide a plain text file (UTF-8 recommended)."
        )

def extract_text_from_file(contents: bytes, filename: str) -> str:
    """Извлекает текст из файлов разных форматов (txt, doc, docx, pdf, odt)."""
    file_extension = os.path.splitext(filename.lower())[1]

    try:
        if file_extension in ['.txt', '.text']:
            return decode_file_contents(contents)

        elif file_extension in ['.doc', '.docx']:
            # Для DOCX используем python-docx напрямую из байтов
            if file_extension == '.docx':
                docx_obj = docx.Document(BytesIO(contents))
                text_parts = []
                for paragraph in docx_obj.paragraphs:
                    text_parts.append(paragraph.text)
                return '\n'.join(text_parts)
            else:  # .doc (старый формат)
                raise HTTPException(
                    status_code=400,
                    detail="DOC format is not fully supported. Please convert to DOCX or TXT."
                )

        elif file_extension == '.pdf':
            pdf_reader = PyPDF2.PdfReader(BytesIO(contents))
            text_parts = []
            for page in pdf_reader.pages:
                text_parts.append(page.extract_text())
            return '\n'.join(text_parts)

        elif file_extension == '.odt':
            doc = Document(BytesIO(contents))
            body = doc.body
            text_parts = []
            # Извлекаем текст из параграфов
            for paragraph in body.get_elements("text:p"):
                text = paragraph.text
                if text and text.strip():
                    text_parts.append(text.strip())

            # Извлекаем текст из заголовков
            for heading in body.get_elements("text:h"):
                text = heading.text
                if text and text.strip():
                    text_parts.append(text.strip())

            return '\n'.join(text_parts)

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file_extension}. Supported formats: txt, docx, pdf, odt."
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file {filename}: {str(e)}"
        )

app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.post("/analyze")
async def analyze_sentiment(
    text: str = Form(None),
    file: UploadFile = File(None)
):
    # Получаем текст из запроса или файла
    if file:
        contents = await file.read()
        text_content = extract_text_from_file(contents, file.filename)
    elif text is not None:
        text_content = text
    else:
        raise HTTPException(
            status_code=400,
            detail="Either text in request body or a file must be provided"
        )

    prompt = f"""Analyze the sentiment of the text. Reply with one word: positive, negative, neutral.
On the next line, write the confidence from 0.0 to 1.0.

Text: {text_content}"""
    try:
        response = ollama.chat(
            model='gemma3:12b',
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0},
           
        )
        content = response['message']['content'].strip()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error communicating with Ollama: {str(e)}"
        )

    lines = content.split('\n')
    if len(lines) < 2:
        raise HTTPException(status_code=500, detail="Invalid LLM response format")

    sentiment = lines[0].lower()
    try:
        confidence = float(lines[1])
    except ValueError:
        raise HTTPException(status_code=500, detail="Invalid confidence value")

    if sentiment not in ["positive", "negative", "neutral"]:
        raise HTTPException(status_code=500, detail="Invalid sentiment label")
    if not (0 <= confidence <= 1):
        raise HTTPException(status_code=500, detail="Confidence must be between 0 and 1")

    return {"sentiment": sentiment, "confidence": confidence}


@app.post("/summarize")
async def summarize_text(
    text: str = Form(None),
    file: UploadFile = File(None)
):
    """
    Endpoint for text summarization. Accepts either text in request body or a text file.
    """
    # Получаем текст из запроса или файла
    if file:
        contents = await file.read()
        text_content = extract_text_from_file(contents, file.filename)
    elif text is not None:
        text_content = text
    else:
        raise HTTPException(
            status_code=400,
            detail="Either text in request body or a file must be provided"
        )

    # Create summarization prompt
    prompt = f"""Create a concise summary of the following text, extracting only the key information.
Keep the summary to 3–5 sentences. Focus on main facts, important details, and essential conclusions.
Omit minor details, examples, and repetitions.

Original text:
{text_content}

Summary (3–5 sentences):"""

    try:
        response = ollama.chat(
            model='gemma3:12b',
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.3, 'num_predict': 300}  # max_tokens аналог
        )
        summary = response['message']['content'].strip()

        # Validate summary
        if not summary or len(summary) < 10:
            raise HTTPException(
                status_code=500,
                detail="Generated summary is too short or empty"
            )

        return {
            "original_text_length": len(text_content),
            "summary_length": len(summary),
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during summarization: {str(e)}"
        )
