from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import ollama


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
        text_content = decode_file_contents(contents)
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
    # Get text either from request or from uploaded file
    if file:
        contents = await file.read()
        text_content = decode_file_contents(contents)
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
