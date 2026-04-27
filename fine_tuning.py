import ollama
import requests
import json

# Настройки
OLLAMA_API_URL = "http://localhost:11434/api/generate"  
MODEL_NAME = "gemma3:1b"

# Примеры для «тонкой настройки» (few‑shot)
TRAINING_EXAMPLES = [
    ("Текст: Отличный сервис, очень доволен!", "Ответ: positive"),
    ("Текст: Ужасно, никогда больше не воспользуюсь", "Ответ: negative"),
    ("Текст: Просто обычный день, ничего особенного", "Ответ: neutral")
]

def query_ollama(prompt, model=MODEL_NAME):
    """Отправляет запрос к локальному Ollama API"""
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 20
        }
    }

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json=data,  
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip()
        else:
            print(f"Ошибка API: {response.status_code}")
            print(f"Текст ошибки: {response.text}")
            return "Ошибка ответа модели"

    except requests.exceptions.ConnectionError:
        return "Ошибка: не удалось подключиться к Ollama. Проверьте, запущен ли 'ollama serve'"
    except Exception as e:
        return f"Неизвестная ошибка: {e}"

def classify_sentiment(text):
    """Классифицирует тональность текста с учётом примеров"""
    # Собираем полную подсказку
    full_prompt = "\n".join([f"{ex[0]}\n{ex[1]}" for ex in TRAINING_EXAMPLES])
    full_prompt += f"\n\nТекст: {text}\nОтвет:"
    return query_ollama(full_prompt)

# Тестирование
if __name__ == "__main__":
    test_texts = [
        "Я очень рад, что попробовал этот продукт!",
        "Это было худшее решение в моей жизни.",
        "Погода сегодня обычная, ничего необычного."
    ]
    print("Результаты классификации:\n")
    for text in test_texts:
        result = classify_sentiment(text)
        print(f"Текст: {text}")
        print(f"Результат: {result}")
        print("-" * 50)
