import import_ipynb
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Literal
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
import uvicorn
from Neoflex_project import extract_text_from_file

# Создаем модель запроса для агента
class AgentRequest(BaseModel):
    task: Literal["sentiment", "summarize"]
    text: str = None
    file: UploadFile = None

# Инициализируем приложение
app = FastAPI()

# Создаем клиента для взаимодействия с существующими эндпоинтами
client = TestClient(app)

@app.post("/agent")
async def process_agent_request(request: AgentRequest):
    try:
        # Проверка входных данных
        if not request.text and not request.file:
            raise HTTPException(status_code=400, detail="Необходимо предоставить текст или файл")

        # Обработка текста или файла
        if request.file:
            contents = await request.file.read()
            text_content = extract_text_from_file(contents, request.file.filename)
        else:
            text_content = request.text

        # Вызываем соответствующий эндпоинт
        if request.task == "sentiment":
            response = client.post(
                "/analyze",
                data={"text": text_content}
            )
            result = response.json()
            return {
                "status": "success",
                "task": "sentiment_analysis",
                "result": {
                    "sentiment": result.get("sentiment"),
                    "confidence": result.get("confidence")
                }
            }

        elif request.task == "summarize":
            response = client.post(
                "/summarize",
                data={"text": text_content}
            )
            result = response.json()
            return {
                "status": "success",
                "task": "text_summarization",
                "result": {
                    "summary": result.get("summary"),
                    "original_length": result.get("original_text_length"),
                    "summary_length": result.get("summary_length")
                }
            }

        else:
            raise HTTPException(status_code=400, detail="Неверный тип задачи")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

