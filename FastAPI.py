from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import io

app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.post("/analyze")
async def analyze_sentiment(request: TextRequest= None, file: UploadFile = File(None)):
    prompt = f"""Analyze the sentiment of the text. Reply with one word: positive, negative, neutral.
On the next line, write the confidence from 0.0 to 1.0.

Text: {request.text}"""
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    lines = response.choices[0].message.content.strip().split('\n')
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
async def summarize_text(request: TextRequest = None, file: UploadFile = File(None)):
    """
    Endpoint for text summarization. Accepts either text in request body or a text file.
    """
    # Get text either from request or from uploaded file
    if file:
        contents = await file.read()
        try:
            text = contents.decode('utf-8')
        except UnicodeDecodeError:
            try:
                text = contents.decode('cp1251')  # Windows-1251 encoding
            except UnicodeDecodeError:
                raise HTTPException(
                    status_code=400, 
                    detail="Unable to decode file. Please ensure it's a text file with UTF-8 or Windows-1251 encoding."
                )
    elif request and request.text:
        text = request.text
    else:
        raise HTTPException(
            status_code=400,
            detail="Either text in request body or a file must be provided"
        )
    
    # Check text length
    if len(text) > 4000:
        raise HTTPException(
            status_code=400,
            detail="Text is too long (max 4000 characters). Please provide shorter text or split it into parts."
        )
    
    # Create summarization prompt
    prompt = f"""Create a concise summary of the following text, extracting only the key information.
Keep the summary to 3–5 sentences. Focus on main facts, important details, and essential conclusions.
Omit minor details, examples, and repetitions.

Original text:
{text}

Summary (3–5 sentences):"""
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Slightly higher for more natural summarization
            max_tokens=300
        )
        
        summary = response.choices[0].message.content.strip()
        
        # Validate summary
        if not summary or len(summary) < 10:
            raise HTTPException(
                status_code=500,
                detail="Generated summary is too short or empty"
            )
            
        return {
            "original_text_length": len(text),
            "summary_length": len(summary),
            "summary": summary
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during summarization: {str(e)}"
        )
