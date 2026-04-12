from pydantic import BaseModel, Field
from typing import Literal

class SentimentAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    original_text_length: int

class SummaryResult(BaseModel):
    summary: str
    original_text_length: int
    summary_length: int
    key_points: list[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    sentiment_analysis: SentimentAnalysis
    summary_result: SummaryResult
    processing_time: float



import time
from fastapi import HTTPException

def analyze_sentiment_tool(text: str) -> SentimentAnalysis:
    """Инструмент для анализа тональности текста"""
    # Используем существующий промт из /analyze endpoint
    prompt = f"""
## ROLE
Professional sentiment analysis expert

## GOAL
Determine sentiment and confidence

## INPUT
Text: {text}

## OUTPUT FORMAT
sentiment (positive/negative/neutral)
confidence (0.0–1.0)

## JSON OUTPUT
{{"sentiment": "positive|negative|neutral", "confidence": 0.0-1.0}}
"""

    try:
        response = ollama.chat(
            model='gemma3:12b',
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0}
        )
        raw_output = response['message']['content'].strip()

        # Извлекаем JSON
        match = re.search(r'\{.*\}$', raw_output, re.DOTALL)
        if match:
            json_str = match.group(0)
            data = json.loads(json_str)
            return SentimentAnalysis(**data, original_text_length=len(text))

        raise HTTPException(status_code=500, detail="Failed to extract sentiment JSON")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis error: {str(e)}")

def summarize_tool(text: str) -> SummaryResult:
    """Инструмент для создания краткого содержания"""
    prompt = f"""
## ROLE
Experienced editor and text analyst

## GOAL
Create concise 3–5 sentence summary

## INPUT
Original text: {text}

## INSTRUCTIONS
1. Extract key facts and main ideas
2. Remove repetitions and minor details
3. Preserve critical numbers and names
4. Use clear language

## OUTPUT FORMAT
Summary text (3–5 sentences)

## JSON OUTPUT
{{"summary": "string", "original_text_length": number, "summary_length": number, "key_points": []}}
"""

    try:
        response = ollama.chat(
            model='gemma3:12b',
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.3, 'num_predict': 300}
        )
        raw_output = response['message']['content'].strip()

        match = re.search(r'\{.*\}$', raw_output, re.DOTALL)
        if match:
            json_str = match.group(0)
            data = json.loads(json_str)
            return SummaryResult(**data)

        raise HTTPException(status_code=500, detail="Failed to extract summary JSON")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization error: {str(e)}")



import logging
from datetime import datetime
from collections import defaultdict

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextAnalysisAgent:
    def __init__(self):
        self.sentiment_retriever = HybridRetriever()
        self.summarize_retriever = HybridRetriever()

        # Политики использования
        self.file_formats_allowed = {'.txt', '.docx', '.pdf', '.odt'}
        self.max_file_size_mb = 10
        self.max_text_length = 5000
        self.request_quota = defaultdict(int)  # квота запросов по пользователям
        self.quota_limit = 100  # запросов в час

        # Инициализация ретриверов с примерами
        texts_sentiment = [doc["text"] for doc in sentiment_documents]
        texts_summarize = [doc["text"] for doc in summarize_documents]

        self.sentiment_retriever.add_documents(texts_sentiment)
        self.summarize_retriever.add_documents(texts_summarize)

    def validate_request(self, user_id: str, file_extension: str = None, text_length: int = 0) -> bool:
        """Валидация запроса по политикам"""
        # Проверка формата файла
        if file_extension and file_extension.lower() not in self.file_formats_allowed:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file_extension}. Allowed: {self.file_formats_allowed}"
            )

        # Проверка длины текста
        if text_length > self.max_text_length:
            raise HTTPException(
                status_code=400,
                detail=f"Text too long: {text_length} > {self.max_text_length} characters"
            )

        # Проверка квоты запросов
        current_time = datetime.now()
        hour_key = f"{user_id}_{current_time.hour}"
        self.request_quota[hour_key] += 1

        if self.request_quota[hour_key] > self.quota_limit:
            raise HTTPException(
                status_code=429,
                detail=f"Request quota exceeded: {self.quota_limit} per hour"
            )

        return True

    async def process_text(self, text: str, user_id: str) -> AnalysisResult:
        """Основной метод обработки текста"""
        start_time = time.time()

        # Валидация запроса
        self.validate_request(user_id, text_length=len(text))

        logger.info(f"Processing text for user {user_id}, length: {len(text)}")

        try:
            # Поиск релевантных примеров для обоих задач
            sentiment_examples = self.sentiment_retriever.retrieve(text, k=2)
            summary_examples = self.summarize_retriever.retrieve(text, k=1)

            # Параллельная обработка
            sentiment_task = analyze_sentiment_tool(text)
            summary_task = summarize_tool(text)

            sentiment_result = await sentiment_task
            summary_result = await summary_task

            processing_time = time.time() - start_time

            # Логирование для мониторинга
            logger.info(
                f"Completed analysis for user {user_id}. "
                f"Sentiment: {sentiment_result.sentiment}, "
                f"Summary length: {summary_result.summary_length}"
            )

            return AnalysisResult(
                sentiment_analysis=sentiment_result,
                summary_result=summary_result,
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"Error processing text for user {user_id}: {str(e)}")
            raise


    async def process_file(self, file: UploadFile, user_id: str) -> AnalysisResult:
        """Обработка файла"""
        # Проверка размера файла
        contents = await file.read()
        file_size_mb = len(contents) / (1024 * 1024)

        if file_size_mb > self.max_file_size_mb:
            raise HTTPException(
                status_code=400,
                detail=f"File too large: {file_size_mb:.2f} MB > {self.max_file_size_mb} MB"
            )

        # Извлечение текста
        text_content = extract_text_from_file(contents, file.filename)

        # Валидация длины текста
        if len(text_content) > self.max_text_length:
            raise HTTPException(
                status_code=400,
                detail=f"Extracted text too long: {len(text_content)} > {self.max_text_length} characters"
            )

        # Обработка текста
        return await self.process_text(text_content, user_id)

    def update_knowledge_base(self, original_text: str, result: AnalysisResult):
        """Обновление базы знаний ретривера (опционально)"""
        if result.sentiment_analysis.confidence > 0.8:
            # Добавляем в ретривер тональности
            example = {
                "text": original_text,
                "metadata": {
                    "type": "sentiment",
                    "label": result.sentiment_analysis.sentiment,
                    "confidence": result.sentiment_analysis.confidence
                }
            }
            self.sentiment_retriever.add_documents([original_text])

        if (result.summary_result.original_text_length > 0 and
            result.summary_result.summary_length > 0):
            # Добавляем в ретривер суммаризации
            example = {
                "text": original_text,
                "summary": result.summary_result.summary,
                "metadata": {
                    "type": "summarization",
                    "original_length": result.summary_result.original_text_length,
                    "summary_length": result.summary_result.summary_length
                }
            }
            self.summarize_retriever.add_documents([original_text])


app = FastAPI()

agent = TextAnalysisAgent()

@app.post("/analyze-and-summarize")
async def analyze_and_summarize(
    text: str = Form(None),
    file: UploadFile = File(None),
    user_id: str = Form(...)  # обязательный параметр user_id
):
    """Единый эндпоинт для анализа тональности и суммаризации"""
    try:
        if file:
            result = await agent.process_file(file, user_id)
        elif text is not None:
            result = await agent.process_text(text, user_id)
        else:
            raise HTTPException(
                status_code=400,
                detail="Either text or a file must be provided"
            )

        # Опциональное обновление базы знаний
        if text:
            agent.update_knowledge_base(text, result)
        elif file:
            contents = await file.read()
            text_content = extract_text_from_file(contents, file.filename)
            agent.update_knowledge_base(text_content, result)

        return result

    except Exception as e:
        logger.error(f"Error in /analyze-and-summarize for user {user_id}: {str(e)}")
        raise


#periodic quality control
import random

def periodic_quality_check():
    """Выборка 5 % случайных результатов для ручной проверки"""
    # Логика выборки из логов или БД
    sample_size = len(agent.request_quota) // 20
    random_samples = random.sample(list(agent.request_quota.keys()), sample_size)

    for sample in random_samples:
        # Отправка на ручную проверку (например, в очередь задач)
        print(f"Need manual review: {sample}")

