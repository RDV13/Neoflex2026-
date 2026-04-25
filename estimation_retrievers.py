import numpy as np
from typing import List, Dict, Any

# === МЕТРИКИ ОЦЕНКИ ===
def calculate_precision_at_k(retrieved_ids: List[int], relevant_ids: List[int], k: int) -> float:
    retrieved_at_k = retrieved_ids[:k]
    if not retrieved_at_k:
        return 0.0
    relevant_retrieved = len(set(retrieved_at_k) & set(relevant_ids))
    return relevant_retrieved / k

def calculate_recall_at_k(retrieved_ids: List[int], relevant_ids: List[int], k: int) -> float:
    retrieved_at_k = retrieved_ids[:k]
    if not relevant_ids:
        return 0.0
    relevant_retrieved = len(set(retrieved_at_k) & set(relevant_ids))
    return relevant_retrieved / len(relevant_ids)

def calculate_mrr(retrieved_lists: List[List[int]], relevant_lists: List[List[int]]) -> float:
    reciprocal_ranks = []
    for retrieved, relevant in zip(retrieved_lists, relevant_lists):
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                reciprocal_ranks.append(1 / rank)
                break
        else:
            reciprocal_ranks.append(0)
    return np.mean(reciprocal_ranks)

# === ОЦЕНКА РЕТРИВЕРА ===
def evaluate_retriever(retriever, test_queries: List[Dict], k: int = 3) -> Dict[str, Any]:
    precision_scores = []
    recall_scores = []
    retrieveddocument_lists = []
    relevantdocument_lists = []

    for query_data in test_queries:
        query = query_data["query"]
        relevant_ids = query_data["relevant_document_ids"]

        # Получаем результаты ретривера
        retrieved_results = retriever.retrieve(query, k=k)
        retrieved_ids = [doc.get("id", i) for i, doc in enumerate(retrieved_results)]

        retrieveddocument_lists.append(retrieved_ids)
        relevantdocument_lists.append(relevant_ids)

        # Считаем метрики для этого запроса
        precision = calculate_precision_at_k(retrieved_ids, relevant_ids, k)
        recall = calculate_recall_at_k(retrieved_ids, relevant_ids, k)

        precision_scores.append(precision)
        recall_scores.append(recall)

    # Агрегируем результаты
    mrr = calculate_mrr(retrieveddocument_lists, relevantdocument_lists)

    return {
        'precision@k': np.mean(precision_scores),
        'recall@k': np.mean(recall_scores),
        'mrr': mrr,
        'n_queries': len(test_queries)
    }

# === ТЕСТОВЫЕ ДАННЫЕ ===
# Тестовые запросы для sentiment analysis
sentiment_test_queries = [
    {
        "query": "Очень доволен покупкой, всё отлично!",
        "relevant_document_ids": [1, 2]
    },
    {
        "query": "Ужасно, никогда больше не куплю",
        "relevant_document_ids": [3, 4]
    },
    {
        "query": "Обычный товар, ничего особенного",
        "relevant_document_ids": [5, 6]
    }
]

# Тестовые запросы для summarization
summarize_test_queries = [
    {
        "query": "Вчера прошёл важный форум по ИИ с участием 1000 специалистов",
        "relevant_document_ids": [7]
    },
    {
        "query": "Конференция в Сан-Франциско собрала 5000 участников",
        "relevant_document_ids": [8]
    }
]

# === ЗАГЛУШКИ РЕТРИВЕРОВ ===
class SentimentRetriever:
    def retrieve(self, query: str, k: int):
        # Имитирует поиск документов для sentiment analysis
        return [{"id": i} for i in range(1, k + 1)]

class SummarizeRetriever:
    def retrieve(self, query: str, k: int):
        # Имитирует поиск документов для summarization
        return [{"id": i + 5} for i in range(k)]

# === ОСНОВНОЙ КОД ===
if __name__ == "__main__":
    # Создаём экземпляры ретриверов
    sentiment_retriever = SentimentRetriever()
    summarize_retriever = SummarizeRetriever()

    # Оцениваем sentiment ретривер
    print("🔎 Оценка sentiment ретривера...")
    sentiment_results = evaluate_retriever(sentiment_retriever, sentiment_test_queries, k=2)
    print(f"Sentiment Retriever: Precision@2 = {sentiment_results['precision@k']:.3f}, "
          f"Recall@2 = {sentiment_results['recall@k']:.3f}, MRR = {sentiment_results['mrr']:.3f}")


    # Оцениваем summarization ретривер
    print("\n🔎 Оценка summarization ретривера...")
    summarize_results = evaluate_retriever(summarize_retriever, summarize_test_queries, k=2)
    print(f"Summarization Retriever: Precision@2 = {summarize_results['precision@k']:.3f}, "
          f"Recall@2 = {summarize_results['recall@k']:.3f}, MRR = {summarize_results['mrr']:.3f}")
