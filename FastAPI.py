from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import ollama
import docx  # python-docx
import PyPDF2
from odfdo import Document as OdfDocument
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

    prompt = f"""
## ROLE
You are a professional sentiment analysis expert. Your task is to accurately determine the emotional tone of text and assess confidence in your judgment.

## GOAL
Determine the sentiment of the text (positive, negative, neutral) and provide a confidence score (0.0 to 1.0).

## TASKS
1. Read the input text carefully.
2. Identify the overall emotional tone.
3. Evaluate confidence in the classification.
4. Provide output in the strictly defined format.

## CONTEXT
The text may be in English or Russian, contain informal language, slang, emojis, and typos. Analyze the overall mood, not individual words.

## INPUT DATA
Text for analysis:
---
{text_content}
---

## ANALYSIS INSTRUCTIONS
1. Positive sentiment: expresses joy, approval, gratitude, enthusiasm.
2. Negative sentiment: contains criticism, dissatisfaction, disappointment, anger.
3. Neutral sentiment: states facts without emotional coloring, contains technical information.


## OUTPUT FORMAT
Strictly follow this format:
- First line: one word — sentiment (`positive`, `negative`, or `neutral`).
- Second line: floating‑point number — confidence (from `0.0` to `1.0`, one decimal place).

## EXAMPLES
Example 1:
Text: "Excellent service, very satisfied!"
Response:
positive
0.9

Example 2:
Text: "Nothing special, just an ordinary day"
Response:
neutral
0.7

Example 3:
Text: "Terrible, I'll never use it again"
Response:
negative
0.8

## TEXT FOR ANALYSIS
---
{text_content}
---
"""
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
    prompt = f"""
## ROLE
You are an experienced editor and text analyst. Your task is to create concise, informative summaries of any texts while preserving key information.

## GOAL
Create a structured summary of 3–5 sentences that conveys main ideas and important details without extra information.

## TASKS
1. Read and analyze the full text.
2. Extract key facts, main ideas, and critical conclusions.
3. Eliminate examples, repetitions, and minor details.
4. Formulate 3–5 coherent sentences reflecting the essence of the text.
5. Ensure the summary reads as a standalone piece.

## CONTEXT
The text may contain technical information, news, articles, reviews, or informal communication. The summary must be understandable without reading the original.

## INPUT DATA
Original text:
---
{text_content}
---

## SUMMARY CREATION INSTRUCTIONS
1. Focus on the main topic and key conclusions.
2. Remove repetitions and redundant details.
3. Preserve critical numbers, dates, and names if essential.
4. Use simple, clear language.
5. Maintain logical coherence between sentences.

## QUALITY CRITERIA
- Length: exactly 3–5 sentences.
- Completeness: all key ideas preserved.
- Conciseness: examples and minor details excluded.
- Readability: text is easily comprehensible.

## OUTPUT FORMAT
Provide only the final summary — 3–5 consecutive sentences without additional comments, headings, or formatting.

## EXAMPLE
Original text: "Yesterday, a major technology forum was held in Moscow. It was attended by over 5 000 professionals from different countries. Key topics included artificial intelligence and cybersecurity. Experts discussed AI development prospects for the next 5 years and shared best practices for data protection. The event concluded with the signing of several important agreements between companies."

Summary: "A technology forum took place in Moscow with over 5 000 professionals in attendance. The main topics were artificial intelligence and cybersecurity. Experts reviewed AI development prospects for the next 5 years and discussed data protection methods. The forum resulted in important inter‑company agreements being signed."

## TEXT TO SUMMARIZE
---
{text_content}
---
"""

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
