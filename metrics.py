import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import precision_score, recall_score

def calculate_precision_at_k(retrieved_ids: List[int], relevant_ids: List[int], k: int) -> float:
    """Вычисляет Precision@k: доля релевантных документов среди первых k результатов."""
    retrieved_at_k = retrieved_ids[:k]
    if not retrieved_at_k:
        return 0.0
    relevant_retrieved = len(set(retrieved_at_k) & set(relevant_ids))
    return relevant_retrieved / k

def calculate_recall_at_k(retrieved_ids: List[int], relevant_ids: List[int], k: int) -> float:
    """Вычисляет Recall@k: доля найденных релевантных документов из всех релевантных."""
    retrieved_at_k = retrieved_ids[:k]
    if not relevant_ids:
        return 0.0
    relevant_retrieved = len(set(retrieved_at_k) & set(relevant_ids))
    return relevant_retrieved / len(relevant_ids)

def calculate_mrr(retrieved_lists: List[List[int]], relevant_lists: List[List[int]]) -> float:
    """Вычисляет Mean Reciprocal Rank: средняя величина, обратная рангу первого релевантного документа."""
    reciprocal_ranks = []
    for retrieved, relevant in zip(retrieved_lists, relevant_lists):
        for rank, doc_id in enumerate(retieved, 1):
            if doc_id in relevant:
                reciprocal_ranks.append(1 / rank)
                break
        else:
            reciprocal_ranks.append(0)
    return np.mean(reciprocal_ranks)

def evaluate_llm_task_performance(baseline_results: List[Dict], enhanced_results: List[Dict]) -> Dict[str, float]:
    """Оценивает влияние ретривера на качество работы LLM."""
    # Сравниваем точность классификации тональности
    baseline_correct = sum(1 for r in baseline_results if r['correct'])
    enhanced_correct = sum(1 for r in enhanced_results if r['correct'])

    baseline_accuracy = baseline_correct / len(baseline_results)
    enhanced_accuracy = enhanced_correct / len(enhanced_results)

    return {
        'baseline_accuracy': baseline_accuracy,
        'enhanced_accuracy': enhanced_accuracy,
        'improvement_percentage': (enhanced_accuracy - baseline_accuracy) * 100
    }
