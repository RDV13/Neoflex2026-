from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai

app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.post("/analyze")
async def analyze_sentiment(request: TextRequest):
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
