import import_ipynb
from retriever import HybridRetriever
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Literal
import logging
import time
import re
import json
import hashlib
import logging
from datetime import datetime
from collections import defaultdict
import random
import time

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


sentiment_documents = [
    {"text": "Отличный сервис, очень доволен!", "metadata": {"type": "sentiment", "label": "positive"}},
    {"text": "Ужасно, никогда больше не воспользуюсь", "metadata": {"type": "sentiment", "label": "negative"}},
    {"text": "Потрясающе! Сервис превзошёл все ожидания — быстрая доставка, вежливый персонал, качество на высоте. Обязательно буду рекомендовать друзьям!",
    "metadata": {"type": "sentiment", "label": "positive"}},
    {"text": "Крайне разочарован. Заказал товар неделю назад, до сих пор нет никакой информации. Поддержка не отвечает, деньги потрачены впустую. Никогда больше не буду пользоваться этой компанией.",
    "metadata": {"type": "sentiment", "label": "negative"}},
    {"text": "Получил заказ сегодня. Товар соответствует описанию, упаковка целая. Никаких проблем не возникло, но и особых восторгов тоже нет — просто выполнил свою функцию.",
    "metadata": {"type": "sentiment", "label": "neutral"}},
    {"text": "Огромное спасибо команде за отличную работу! Всё сделали быстро, чётко и с душой. Очень приятно иметь дело с профессионалами. Буду обращаться ещё!",
    "metadata": {"type": "sentiment", "label": "positive"}},
    {"text": "Ужасное обслуживание. Оператор грубил, решение проблемы затянулось на несколько дней. Качество товара тоже оставляет желать лучшего — видно, что сэкономили на материалах.",
    "metadata": {"type": "sentiment", "label": "negative"}},
    {"text": "Amazing experience! The product arrived earlier than expected, and it's even better than I imagined. Great value for money, highly recommend!",
    "metadata": {"type": "sentiment", "label": "positive"}},
    {"text": "Terrible service. Waited 3 hours on hold, then the agent couldn't help me. Product arrived damaged, and the company refuses to replace it. Very disappointed.",
    "metadata": {"type": "sentiment", "label": "negative"}},
    {"text": "Received the item as described. It works as expected, no issues so far. Nothing extraordinary, but it does the job. Average experience overall.",
    "metadata": {"type": "sentiment", "label": "neutral"}},
    {"text": "I'm very impressed with the quality and attention to detail. The team went above and beyond to meet my requirements. Will definitely order again!",
    "metadata": {"type": "sentiment", "label": "positive"}},
    {"text": "Waste of money. The product broke after one week of use. Customer service was unhelpful and rude. I regret this purchase and won't recommend this brand.",
    "metadata": {"type": "sentiment", "label": "negative"}},
    {"text": "The service is acceptable. It took longer than promised, but eventually I got what I ordered. Nothing to complain about, but also nothing to rave about.",
    "metadata": {"type": "sentiment", "label": "neutral"}}
]

# База для суммирования
summarize_documents = [
    {"text": "Вчера в Москве прошёл масштабный технологический форум, собравший более 5 000 специалистов из разных стран. Ключевыми темами мероприятия стали искусственный интеллект и кибербезопасность. Эксперты обсудили перспективы развития ИИ на ближайшие 5 лет и поделились лучшими практиками защиты данных. В рамках форума состоялось подписание нескольких важных соглашений между ведущими технологическими компаниями.",
    "summary": "В Москве прошёл технологический форум с участием 5 000 специалистов. Основные темы — ИИ и кибербезопасность. Обсуждены перспективы развития ИИ на 5 лет, подписаны межкорпоративные соглашения.",
    "metadata": {"type": "summarization", "domain": "news", "length_original": 342, "length_summary": 168}},
    {"text": "Купил этот ноутбук 3 месяца назад для работы и остался очень доволен. Процессор Intel i7 справляется с многозадачностью без проблем, 16 ГБ оперативной памяти хватает для всех задач. Экран яркий, цвета точные, что важно для дизайнера. Единственный минус — батарея держит всего 4 часа при активной работе. В целом, отличное соотношение цены и качества, рекомендую коллегам.",
    "summary": "Ноутбук с процессором Intel i7 и 16 ГБ ОЗУ хорошо подходит для работы, особенно для дизайна (точный цветопередача). Минус — батарея на 4 часа. Хорошее соотношение цены и качества.",
    "metadata": {"type": "summarization", "domain": "reviews", "length_original": 298, "length_summary": 158}},
    {"text": "Чтобы создать эффективное резюме, следуйте этим шагам: 1) Укажите ваше имя и контактную информацию в верхней части документа. 2) Кратко опишите ваш профессиональный опыт (не более 3–5 предложений). 3) Перечислите ключевые навыки, релевантные для желаемой должности. 4) Укажите образование и любые сертификаты. 5) При необходимости добавьте раздел «Достижения» с 2–3 пунктами. Старайтесь уместить всё на одной странице.",
    "summary": "Для создания резюме: укажите имя и контакты, кратко опишите опыт (3–5 предложений), перечислите ключевые навыки, образование и сертификаты. При необходимости добавьте 2–3 достижения. Уместите на одной странице.",
    "metadata": {"type": "summarization", "domain": "instructions", "length_original": 356, "length_summary": 182}},
    {"text": "A major technology conference was held in San Francisco last week, attracting over 7 000 attendees from 50 countries. Keynote speakers discussed advancements in renewable energy and electric vehicles. The event featured product launches from leading tech companies, including a new solar panel with 25 % higher efficiency. Several partnerships were announced between automotive and energy firms.",
    "summary": "San Francisco hosted a tech conference with 7 000+ attendees. Focus on renewable energy and EVs. New high‑efficiency solar panel launched; multiple automotive‑energy partnerships announced.",
    "metadata": {"type": "summarization", "domain": "news", "length_original": 289, "length_summary": 147}},
    {"text": "I've been using this wireless earbuds for two months now, and they've exceeded my expectations. The sound quality is excellent, with clear highs and deep bass. Battery life lasts 6 hours on a single charge, and the case provides three additional full charges. The only drawback is the touch controls are sometimes unresponsive. Overall, a great purchase for the price point.",
    "summary": "Wireless earbuds offer excellent sound (clear highs, deep bass) and 6‑hour battery (plus 3 charges from case). Minor issue: unresponsive touch controls. Good value overall.",
    "metadata": {"type": "summarization", "domain": "reviews", "length_original": 278, "length_summary": 139}},
    {"text": "To create an effective summary, follow these steps: 1) Read the original text thoroughly to understand the main points. 2) Identify the key ideas and supporting details. 3) Remove examples, repetitions, and minor details. 4) Combine related ideas into concise sentences. 5) Ensure the summary is 3–5 sentences long and preserves the original meaning. 6) Proofread for clarity and coherence.",
    "summary": "Create a summary by: reading thoroughly, identifying key ideas, removing extras, combining related points, keeping 3–5 sentences, and proofreading. Preserve original meaning.",
    "metadata": {"type": "summarization", "domain": "instructions", "length_original": 267, "length_summary": 143}},
    {"text": "Современные методы радиоволновой диагностики неоднородностей в ионосфере: теоретические основы и экспериментальные результаты\n\nИоносфера Земли представляет собой сложную плазменную среду, характеризующуюся высокой пространственно‑временной изменчивостью параметров. В последние десятилетия особый интерес вызывает изучение мелкомасштабных неоднородностей электронной концентрации, которые оказывают существенное влияние на распространение радиоволн различных диапазонов.\n\nТеоретическая модель, описывающая взаимодействие радиоволн с ионосферными неоднородностями, основывается на решении уравнения переноса излучения в случайно‑неоднородной среде. При этом учитываются следующие ключевые факторы:\n1) анизотропия флуктуаций электронной концентрации;\n2) зависимость коэффициента рассеяния от частоты радиоволны;\n3) эффекты многократного рассеяния;\n4) влияние геомагнитного поля на распространение волн.\n\nДля экспериментальной верификации теоретических предсказаний был проведён цикл измерений с использованием сети ионозондов вертикального зондирования и системы некогерентного рассеяния (ИСЗ «Ионосфера‑М»). Эксперименты проводились в период высокой солнечной активности (F10.7 > 180), что обеспечивало наличие интенсивных неоднородностей.\n\nОсновные результаты измерений:\n* зарегистрированы неоднородности с масштабами от 100 м до 50 км;\n* выявлена корреляция между интенсивностью неоднородностей и геомагнитной активностью (коэффициент корреляции r = 0.87);\n* обнаружено аномальное усиление флуктуаций фазы на частотах 10–30 МГц;\n* измерены вертикальные скорости перемещения неоднородностей (диапазон 50–200 м/с).\n\nОсобое внимание уделено анализу эффектов замирания сигнала (fading) при спутниковой связи. На основе полученных данных разработана методика коррекции ошибок, учитывающая:\n* статистические характеристики замираний;\n* пространственную структуру неоднородностей;\n* частотную зависимость коэффициента ослабления.\n\nПрактическая значимость результатов заключается в возможности:\n* повышения точности систем спутниковой навигации (GPS/ГЛОНАСС) на 15–20 % в условиях возмущённой ионосферы;\n* улучшения качества коротковолновой связи в полярных регионах;\n* оптимизации работы систем загоризонтной радиолокации.\n\nДальнейшие исследования планируется направить на изучение динамики неоднородностей в условиях различных геомагнитных бурь и разработку адаптивных алгоритмов компенсации ионосферных искажений.",
    "summary": "Исследование посвящено диагностике неоднородностей в ионосфере. Разработана теоретическая модель взаимодействия радиоволн с учётом анизотропии флуктуаций, частоты волны, многократного рассеяния и геомагнитного поля. Эксперименты с ионозондами и системой некогерентного рассеяния выявили неоднородности (100 м–50 км), их связь с геомагнитной активностью (r = 0.87), аномальное усиление флуктуаций фазы (10–30 МГц) и скорости перемещения (50–200 м/с). Разработана методика коррекции ошибок для спутниковой связи. Практическая польза: повышение точности GPS/ГЛОНАСС на 15–20 %, улучшение КВ‑связи в полярных регионах и оптимизация загоризонтной радиолокации.",
    "metadata": {
        "type": "summarization",
        "domain": "radio_physics",
        "topic": "ionospheric_irregularities",
        "length_original": 1384,
        "length_summary": 498,
        "key_elements_preserved": [
            "масштабы неоднородностей (100 м–50 км)",
            "коэффициент корреляции (r = 0.87)",
            "диапазон частот (10–30 МГц)",
            "скорости перемещения (50–200 м/с)",
            "повышение точности GPS/ГЛОНАСС (15–20 %)"
        ]}}
]


def analyze_sentiment_tool(text: str) -> SentimentAnalysis:
    """Инструмент для анализа тональности текста"""
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
    user_id: str = Form("anonymous")
    #user_id: str = Form(...)  # обязательный параметр user_id
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

def periodic_quality_check():
    """Выборка 5 % случайных результатов для ручной проверки"""
    # Логика выборки из логов или БД
    sample_size = len(agent.request_quota) // 20
    random_samples = random.sample(list(agent.request_quota.keys()), sample_size)

    for sample in random_samples:
        # Отправка на ручную проверку (например, в очередь задач)
        print(f"Need manual review: {sample}")
