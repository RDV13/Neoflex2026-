from app.retriever import sentiment_retriever, summarize_retriever
from evaluation.test_data import sentiment_test_queries, summarize_test_queries
from evaluation.metrics import calculate_precision_at_k, calculate_recall_at_k, calculate_mrr
from evaluation.visualization import generate_detailed_report, visualize_evaluation_results
from evaluation.optimization import generate_optimization_recommendations

import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import precision_score, recall_score

def evaluate_retriever(retriever, test_queries: List[Dict], k: int = 3) -> Dict[str, Any]:
    """Проводит комплексную оценку ретривера."""
    precision_scores = []
    recall_scores = []
    retrieved_document_lists = []
    relevant_document_lists = []

    for query_data in test_queries:
        query = query_data["query"]
        relevant_ids = query_data["relevant_document_ids"]

        # Получаем результаты ретривера
        retrieved_results = retriever.retrieve(query, k=k)
        retieved_ids = [doc['id'] for doc in retieved_results]  # предполагаем, что у документов есть id

        retieveddocument_lists.append(retieved_ids)
        relevantdocument_lists.append(relevant_ids)

        # Считаем метрики для этого запроса
        precision = calculate_precision_at_k(retieved_ids, relevant_ids, k)
        recall = calculate_recall_at_k(retieved_ids, relevant_ids, k)

        precision_scores.append(precision)
        recall_scores.append(recall)

    # Агрегируем результаты
    mrr = calculate_mrr(retieveddocument_lists, relevantdocument_lists)

    return {
        'precision@k': np.mean(precision_scores),
        'recall@k': np.mean(recall_scores),
        'mrr': mrr,
        'detailed_results': {
            'precision_per_query': precision_scores,
            'recall_per_query': recall_scores,
            'retieved_documents': retieveddocument_lists
        }
    }


def run_llm_performance_evaluation():
    """Запускает оценку влияния ретривера на работу LLM."""
    sentiment_results_baseline = []
    sentiment_results_enhanced = []

    for test_case in sentiment_test_queries:
        text = test_case["query"]
        expected_sentiment = test_case["expected_sentiment"]

        # Базовое выполнение (без ретривера)
        baseline_response = ollama.chat(
            model='gemma3:12b',
            messages=[{'role': 'user', 'content': f"Determine sentiment: {text}"}],
            options={'temperature': 0}
        )
        baseline_sentiment = parse_sentiment_response(baseline_response['message']['content'])
        sentiment_results_baseline.append({
            'correct': baseline_sentiment == expected_sentiment,
            'response': baseline_response
        })

        # Выполнение с ретривером
        relevant_examples = sentiment_retriever.retrieve(text, k=2)
        enhanced_prompt = create_enhanced_prompt(text, relevant_examples)
        enhanced_response = ollama.chat(
            model='gemma3:12b',
            messages=[{'role': 'user', 'content': enhanced_prompt}],
            options={'temperature': 0}
        )
        enhanced_sentiment = parse_sentiment_response(enhanced_response['message']['content'])
        sentiment_results_enhanced.append({
            'correct': enhanced_sentiment == expected_sentiment,
            'response': enhanced_response
        })

    # Оцениваем улучшение
    impact_results = evaluate_llm_task_performance(
        sentiment_results_baseline,
        sentiment_results_enhanced
    )

    return impact_results



# Оцениваем sentiment ретривер
print("Evaluating Sentiment Retriever...")
sentiment_evaluation = evaluate_retriever(sentiment_retriever, sentiment_test_queries, k=3)

# Оцениваем summarization ретривер
print("Evaluating Summarization Retriever...")
summarize_evaluation = evaluate_retriever(summarize_retriever, summarize_test_queries, k=3)


# Оцениваем влияние на LLM
print("Evaluating LLM Performance Impact...")
llm_impact = run_llm_performance_evaluation()



import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def save_evaluation_results(sentiment_eval, summarize_eval, llm_impact, filename: str):
    """Сохраняет результаты оценки в JSON."""
    results = {
        'sentiment_retriever': sentiment_eval,
        'summarize_retriever': summarize_eval,
        'llm_performance_impact': llm_impact,
        'evaluation_timestamp': str(datetime.now())
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Результаты сохранены в {filename}")

def visualize_evaluation_results(sentiment_eval, summarize_eval, llm_impact):
    """Визуализирует результаты оценки."""
    # Создаём фигуру с подграфиками
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Результаты оценки ретриверов', fontsize=16)

    # 1. Precision@k и Recall@k для sentiment ретривера
    metrics = ['precision@k', 'recall@k']
    values_sentiment = [sentiment_eval[m] for m in metrics]
    axes[0, 0].bar(metrics, values_sentiment, color='skyblue')
    axes[0, 0].set_title('Sentiment Retriever: Precision@k & Recall@k')
    axes[0, 0].set_ylim(0, 1)

    # 2. MRR для sentiment ретривера
    axes[0, 1].bar(['MRR'], [sentiment_eval['mrr']], color='lightgreen')
    axes[0, 1].set_title('Sentiment Retriever: MRR')
    axes[0, 1].set_ylim(0, 1)

    # 3. Precision@k и Recall@k для summarization ретривера
    values_summarize = [summarize_eval[m] for m in metrics]
    axes[1, 0].bar(metrics, values_summarize, color='salmon')
    axes[1, 0].set_title('Summarization Retriever: Precision@k & Recall@k')
    axes[1, 0].set_ylim(0, 1)

    # 4. Влияние на LLM
    llm_metrics = ['baseline_accuracy', 'enhanced_accuracy']
    llm_values = [llm_impact[m] for m in llm_metrics]
    axes[1, 1].bar(llm_metrics, llm_values, color='gold')
    axes[1, 1].set_title('Влияние на точность LLM')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].text(0, llm_impact['baseline_accuracy'] + 0.02,
                   f"{llm_impact['baseline_accuracy']:.2%}", ha='center')
    axes[1, 1].text(1, llm_impact['enhanced_accuracy'] + 0.02,
                   f"{llm_impact['enhanced_accuracy']:.2%}", ha='center')

    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# Запуск сохранения и визуализации
save_evaluation_results(
    sentiment_evaluation,
    summarize_evaluation,
    llm_impact,
    'retriever_evaluation_results.json'
)
visualize_evaluation_results(sentiment_evaluation, summarize_evaluation, llm_impact)




def run_full_evaluation():
    """Полный цикл оценки ретриверов."""
    print("🚀 Запуск полной оценки ретриверов...")

    # Оцениваем sentiment ретривер
    print("🔎 Оцениваем Sentiment Retriever...")
    sentiment_evaluation = evaluate_retriever(sentiment_retriever, sentiment_test_queries, k=3)

    # Оцениваем summarization ретривер
    print("🔎 Оцениваем Summarization Retriever...")
    summarize_evaluation = evaluate_retriever(summarize_retriever, summarize_test_queries, k=3)


    # Оцениваем влияние на LLM
    print("🤖 Оцениваем влияние на LLM...")
    llm_impact = run_llm_performance_evaluation()

    # Сохраняем и визуализируем
    save_evaluation_results(
        sentiment_evaluation,
        summarize_evaluation,
        llm_impact,
        f'retriever_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    )
    visualize_evaluation_results(sentiment_evaluation, summarize_evaluation, llm_impact)


    # Выводим сводку в консоль
    print("\n" + "="*50)
    print("📊 СВОДКА РЕЗУЛЬТАТОВ ОЦЕНКИ")
    print("="*50)
    print(f"Sentiment Retriever:")
    print(f"  Precision@3: {sentiment_evaluation['precision@k']:.3f}")
    print(f"  Recall@3:  {sentiment_evaluation['recall@k']:.3f}")
    print(f"  MRR:       {sentiment_evaluation['mrr']:.3f}")
    print(f"\nSummarization Retriever:")
    print(f"  Precision@3: {summarize_evaluation['precision@k']:.3f}")
    print(f"  Recall@3:  {summarize_evaluation['recall@k']:.3f}")
    print(f"  MRR:       {summarize_evaluation['mrr']:.3f}")
    print(f"\nВлияние на LLM:")
    print(f"  Базовая точность:    {llm_impact['baseline_accuracy']:.1%}")
    print(f"  Точность с ретривером: {llm_impact['enhanced_accuracy']:.1%}")
    print(f"  Улучшение:          {llm_impact['improvement_percentage']:.1f} п.п.")
    print("="*50)

# Запускаем полную оценку
run_full_evaluation()


def analyze_retriever_errors(retriever, test_queries, k: int = 3):
    """Анализирует типы ошибок ретривера."""
    error_analysis = {
        'missing_relevant': [],  # запросы, где не найдены релевантные документы
        'low_confidence': [],   # запросы с низкой уверенностью ретривера
        'wrong_order': []       # релевантные документы не на первых позициях
    }

    for query_data in test_queries:
        query = query_data["query"]
        relevant_ids = query_data["relevant_document_ids"]

        retrieved_results = retriever.retrieve(query, k=k)
        retieved_ids = [doc['id'] for doc in retieved_results]


        # Проверка 1: отсутствуют релевантные документы
        if not any(doc_id in relevant_ids for doc_id in retieved_ids):
            error_analysis['missing_relevant'].append(query)


        # Проверка 2: низкая уверенность (если есть scores)
        if 'score' in retieved_results[0]:
            avg_score = np.mean([doc['score'] for doc in retieved_results])
            if avg_score < 0.3:  # порог уверенности
                error_analysis['low_confidence'].append({
                    'query': query,
                    'avg_score': avg_score
                })

        # Проверка 3: неправильный порядок
        first_relevant_rank = None
        for rank, doc_id in enumerate(retieved_ids, 1):
            if doc_id in relevant_ids:
                first_relevant_rank = rank
                break
        if first_relevant_rank and first_relevant_rank > 1:
            error_analysis['wrong_order'].append({
                'query': query,
                'first_relevant_rank': first_relevant_rank
            })

    return error_analysis



# Запуск анализа ошибок для обоих ретриверов
print("🔎 Анализ ошибок Sentiment Retriever...")
sentiment_errors = analyze_retriever_errors(sentiment_retriever, sentiment_test_queries, k=3)

print("🔎 Анализ ошибок Summarization Retriever...")
summarize_errors = analyze_retriever_errors(summarize_retriever, summarize_test_queries, k=3)


def print_error_analysis(sentiment_errors, summarize_errors):
    """Выводит отчёт по ошибкам ретриверов."""
    print("\n" + "="*60)
    print("🛑 АНАЛИЗ ОШИБОК РЕТРИВЕРОВ")
    print("="*60)

    print("\n🎯 SENTIMENT RETRIEVER:")
    print(f"  • Запросы без релевантных документов: {len(sentiment_errors['missing_relevant'])}")
    if sentiment_errors['missing_relevant']:
        print("    Примеры:")
        for q in sentiment_errors['missing_relevant'][:3]:
            print(f"      - '{q[:50]}...'")

    print(f"  • Запросы с низкой уверенностью: {len(sentiment_errors['low_confidence'])}")
    if sentiment_errors['low_confidence']:
        for item in sentiment_errors['low_confidence'][:2]:
            print(f"    - '{item['query'][:40]}...': уверенность {item['avg_score']:.2f}")

    print(f"  • Релевантные не на первом месте: {len(sentiment_errors['wrong_order'])}")

    print("\n📝 SUMMARIZATION RETRIEVER:")
    print(f"  • Запросы без релевантных документов: {len(summarize_errors['missing_relevant'])}")
    print(f"  • Запросы с низкой уверенностью: {len(summarize_errors['low_confidence'])}")
    print(f"  • Релевантные не на первом месте: {len(summarize_errors['wrong_order'])}")
    print("="*60)

print_error_analysis(sentiment_errors, summarize_errors)



# Генерация подробного отчёта в формате Markdown
def generate_detailed_report(sentiment_eval, summarize_eval, llm_impact,
                          sentiment_errors, summarize_errors, filename: str):
    """Генерирует подробный отчёт в формате Markdown."""

    report = f"""# Отчёт оценки ретриверов

**Дата генерации:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. Основные метрики

### Sentiment Retriever
| Метрика | Значение |
|--------|----------|
| Precision@3 | {sentiment_eval['precision@k']:.3f} |
| Recall@3 | {sentiment_eval['recall@k']:.3f} |
| MRR | {sentiment_eval['mrr']:.3f} |

### Summarization Retriever
| Метрика | Значение |
|--------|----------|
| Precision@3 | {summarize_eval['precision@k']:.3f} |
| Recall@3 | {summarize_eval['recall@k']:.3f} |
| MRR | {summarize_eval['mrr']:.3f} |

## 2. Влияние на LLM

| Параметр | Значение |
|---------|----------|
| Базовая точность | {llm_impact['baseline_accuracy']:.1%} |
| Точность с ретривером | {llm_impact['enhanced_accuracy']:.1%} |
| Улучшение | {llm_impact['improvement_percentage']:.1f} п.п. |

## 3. Анализ ошибок

### Sentiment Retriever
* Запросы без релевантных документов: {len(sentiment_errors['missing_relevant'])}
* Запросы с низкой уверенностью: {len(sentiment_errors['low_confidence'])}
* Релевантные документы не на первом месте: {len(sentiment_errors['wrong_order'])}

### Summarization Retriever
* Запросы без релевантных документов: {len(summarize_errors['missing_relevant'])}
* Запросы с низкой уверенностью: {len(summarize_errors['low_confidence'])}
* Релевантные документы не на первом месте: {len(summarize_errors['wrong_order'])}


## 4. Выводы и рекомендации

"""

    # Добавляем выводы на основе результатов
    if llm_impact['improvement_percentage'] > 5:
        report += "- ✅ Ретривер значительно улучшает работу LLM (+{:.1f} п.п.)\n".format(llm_impact['improvement_percentage'])
        report += "- Рекомендуется использовать ретривер в продакшене\n"
    elif llm_impact['improvement_percentage'] > 0:
        report += "- △ Ретривер даёт небольшое улучшение (+{:.1f} п.п.)\n".format(llm_impact['improvement_percentage'])
        report += "- Рассмотреть оптимизацию ретривера\n"
    else:
        report += "- ⚠️ Ретривер не улучшает или ухудшает результаты\n"
        report += "- Требуется доработка ретривера или промтов\n"

    if len(sentiment_errors['missing_relevant']) > 3:
        report += "- ❗ Много запросов без релевантных документов — расширить базу знаний\n"


    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"Подробный отчёт сохранён в {filename}")

# Генерация отчёта
generate_detailed_report(
    sentiment_evaluation,
    summarize_evaluation,
    llm_impact,
    sentiment_errors,
    summarize_errors,
    'detailed_evaluation_report.md'
)


# Автоматизированная проверка воспроизводимости
def run_reproducibility_test(n_runs: int = 3):
    """Запускает оценку несколько раз для проверки воспроизводимости."""
    results_over_runs = []

    for run in range(n_runs):
        print(f"\n🔁 Запуск {run + 1}/{n_runs} для проверки воспроизводимости...")
        set_reproducibility_seeds()  # фиксируем seed перед каждым запуском


        sentiment_eval = evaluate_retriever(sentiment_retriever, sentiment_test_queries, k=3)
        summarize_eval = evaluate_retriever(summarize_retriever, summarize_test_queries, k=3)
        llm_impact = run_llm_performance_evaluation()

        results_over_runs.append({
            'run': run + 1,
            'sentiment_precision': sentiment_eval['precision@k'],
            'summarize_precision': summarize_eval['precision@k'],
            'llm_improvement': llm_impact['improvement_percentage']
        })

    # Анализируем вариативность
    precisions_sentiment = [r['sentiment_precision'] for r in results_over_runs]
    precisions_summarize = [r['summarize_precision'] for r in results_over_runs]

    variability_sentiment = np.std(precisions_sentiment)
    variability_summarize = np.std(precisions_summarize)

    print("\n📊 РЕЗУЛЬТАТЫ ПРОВЕРКИ ВОСПРОИЗВОДИМОСТИ")
    print(f"• Вариативность Precision@3 (Sentiment): {variability_sentiment:.4f}")
    print(f"• Вариативность Precision@3 (Summarize): {variability_summarize:.4f}")

    if variability_sentiment < 0.05 and variability_summarize < 0.05:
        print("✅ Результаты воспроизводимы (вариативность < 0.05)")
    else:
        print("⚠️  Результаты показывают значительную вариативность — проверьте источники случайности")

# Запускаем проверку воспроизводимости
run_reproducibility_test()



def main_evaluation_pipeline():
    """Полный конвейер оценки ретриверов."""
    print("🚀 ЗАПУСК ПОЛНОГО КОНВЕЙЕРА ОЦЕНКИ РЕТРИВЕРОВ")
    print("=" * 60)

    # Шаг 1. Установка воспроизводимости
    set_reproducibility_seeds()

    # Шаг 2. Оценка ретриверов с разными k
    print("🔍 Запуск оценки для разных значений k...")
    k_results = run_evaluation_with_different_k()

    # Шаг 3. Основная оценка
    print("\n🔎 Запуск основной оценки...")
    full_results = run_full_evaluation()

    # Шаг 4. Анализ ошибок
    print("\n🛑 Анализ ошибок ретриверов...")
    sentiment_errors = analyze_retriever_errors(
        sentiment_retriever, sentiment_test_queries, k=3
    )
    summarize_errors = analyze_retriever_errors(
        summarize_retriever, summarize_test_queries, k=3
    )

    # Шаг 5. Генерация подробного отчёта
    print("\n📝 Генерация подробного отчёта...")
    generate_detailed_report(
        full_results['sentiment_eval'],
        full_results['summarize_eval'],
        full_results['llm_impact'],
        sentiment_errors,
        summarize_errors,
        'detailed_evaluation_report.md'
    )

    # Шаг 6. Проверка воспроизводимости
    print("\n🔁 Проверка воспроизводимости...")
    run_reproducibility_test()

    # Шаг 7. Рекомендации по оптимизации
    print("\n💡 ГЕНЕРАЦИЯ РЕКОМЕНДАЦИЙ...")
    generate_optimization_recommendations(
        full_results,
        sentiment_errors,
        summarize_errors
    )

    print("\n🎉 ОЦЕНКА ЗАВЕРШЕНА!")
    print("=" * 60)
    return full_results

# Запуск полного конвейера
if __name__ == "__main__":
    results = main_evaluation_pipeline()



# Генерация рекомендаций по оптимизации
def generate_optimization_recommendations(full_results, sentiment_errors, summarize_errors):
    """Генерирует рекомендации по улучшению ретриверов на основе результатов оценки."""
    print("\n" + "=" * 50)
    print("💡 РЕКОМЕНДАЦИИ ПО ОПТИМИЗАЦИИ")
    print("=" * 50)

    recommendations = []

    # Анализ результатов LLM
    improvement = full_results['llm_impact']['improvement_percentage']
    if improvement > 5:
        recommendations.append(
            "✅ Ретривер значительно улучшает работу LLM (+{:.1f} п.п.). "
            "Рекомендуется использовать в продакшене.".format(improvement)
        )
    elif improvement > 0:
        recommendations.append(
            "△ Ретривер даёт небольшое улучшение (+{:.1f} п.п.). "
            "Рассмотреть оптимизацию ретривера или промтов.".format(improvement)
        )
    else:
        recommendations.append(
            "⚠ Ретривер не улучшает или ухудшает результаты. "
            "Требуется доработка ретривера, базы знаний или промтов."
        )

    # Анализ ошибок Sentiment Retriever
    missing_sentiment = len(sentiment_errors['missing_relevant'])
    if missing_sentiment > 3:
        recommendations.append(
            f"❗ {missing_sentiment} запросов не нашли релевантных документов. "
            "Расширить базу знаний для sentiment analysis."
        )

    low_confidence_sentiment = len(sentiment_errors['low_confidence'])
    if low_confidence_sentiment > 2:
        recommendations.append(
            f"⚠️ {low_confidence_sentiment} запросов с низкой уверенностью ретривера. "
            "Улучшить эмбеддинги или использовать гибридный поиск."
        )

    # Анализ ошибок Summarization Retriever
    missing_summarize = len(summarize_errors['missing_relevant'])
    if missing_summarize > 2:
        recommendations.append(
            f"❗ {missing_summarize} запросов не нашли релевантных примеров. "
            "Добавить больше примеров summarization в базу знаний."
        )

    # Анализ метрик Precision/Recall
    sent_prec = full_results['sentiment_eval']['precision@k']
    sent_rec = full_results['sentiment_eval']['recall@k']

    if sent_prec < 0.7:
        recommendations.append(
            "⚠️ Низкая Precision@3 ({:.2f}) для Sentiment Retriever. "
            "Оптимизировать ранжирование результатов.".format(sent_prec)
        )
    if sent_rec < 0.6:
        recommendations.append(
            "⚠️ Низкий Recall@3 ({:.2f}) для Sentiment Retriever. "
            "Улучшить покрытие базы знаний.".format(sent_rec)
        )

    # Вывод всех рекомендаций
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")

    print("=" * 50)
