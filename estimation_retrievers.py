import numpy as np
from typing import List, Dict, Any
from metrics import (
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_mrr
)
from retriever import HybridRetriever

# База для анализа тональности
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

def create_test_queries(documents: List[Dict]) -> List[Dict[str, Any]]:
    """Создаёт тестовые запросы из документов."""
    test_queries = []
    for i, doc in enumerate(documents):
        test_queries.append({
            "query": doc["text"],
            "relevant_document_ids": [i]  # сам документ — релевантный
        })
    return test_queries

def evaluate_retriever(
    retriever: HybridRetriever,
    documents: List[Dict],
    k: int = 3
) -> Dict[str, float]:
    """Оценивает ретривер на наборе документов."""
    # Создаём тестовые запросы
    test_queries = create_test_queries(documents)

    # Подготавливаем данные для метрик
    precision_scores = []
    recall_scores = []
    retrieved_lists = []  
    relevant_lists = []

    # Извлекаем тексты документов
    texts = [doc["text"] for doc in documents]


    # Добавляем документы в ретривер
    retriever.add_documents(texts)

    print(f"🔎 Оцениваем ретривер на {len(test_queries)} тестовых запросах...")

    for query_data in test_queries:
        query = query_data["query"]
        relevant_ids = query_data["relevant_document_ids"]

        try:
            # Получаем результаты поиска
            retrieved_results = retriever.retrieve(query, k=k)
            retieved_ids = [result["source_id"] for result in retrieved_results]
        except Exception as e:
            print(f"Ошибка при обработке запроса '{query}': {e}")
            retieved_ids = []

        # Сохраняем для MRR
        retrieved_lists.append(retieved_ids)  
        relevant_lists.append(relevant_ids)

        # Считаем метрики для этого запроса
        if retieved_ids:
            precision = calculate_precision_at_k(retieved_ids, relevant_ids, k)
            recall = calculate_recall_at_k(retieved_ids, relevant_ids, k)
        else:
            precision = 0.0
            recall = 0.0

        precision_scores.append(precision)
        recall_scores.append(recall)

    # Считаем MRR
    mrr = calculate_mrr(retrieved_lists, relevant_lists)

    return {
        'precision@k': np.mean(precision_scores),
        'recall@k': np.mean(recall_scores),
        'mrr': mrr,
        'n_queries': len(test_queries),
        'k': k
    }

if __name__ == "__main__":
    # Инициализируем ретривер
    retriever = HybridRetriever()

    # Оцениваем на sentiment документах
    print("📊 Оценка ретривера для анализа тональности:")
    sentiment_results = evaluate_retriever(retriever, sentiment_documents, k=3)
    for metric, value in sentiment_results.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.3f}")
        else:
            print(f"{metric}: {value}")

    print("\n" + "="*50 + "\n")

    # Оцениваем на summarize документах
    print("📊 Оценка ретривера для суммирования:")
    summarize_results = evaluate_retriever(retriever, summarize_documents, k=3)
    for metric, value in summarize_results.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.3f}")
        else:
            print(f"{metric}: {value}")
