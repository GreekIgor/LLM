"""
Скрипт для тестирования vLLM сервера
Быстрая проверка доступности и работоспособности
"""

import requests
import sys
from typing import Dict, Any

def test_server_health(base_url: str = "http://localhost:8000") -> bool:
    """Проверяет доступность сервера"""
    print("🔍 Проверка доступности vLLM сервера...")
    
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Сервер доступен и работает")
            return True
        else:
            print(f"⚠️  Сервер ответил со статусом: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Не удалось подключиться к серверу")
        print(f"   Убедитесь, что сервер запущен на {base_url}")
        return False
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False


def get_available_models(base_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Получает список доступных моделей"""
    print("\n📋 Получение списка моделей...")
    
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=5)
        response.raise_for_status()
        
        models = response.json()
        print("✅ Доступные модели:")
        
        for model in models.get('data', []):
            print(f"   - {model.get('id')}")
            print(f"     Created: {model.get('created')}")
            print(f"     Owner: {model.get('owned_by')}")
        
        return models
        
    except Exception as e:
        print(f"❌ Ошибка получения моделей: {e}")
        return {}


def test_completion(base_url: str = "http://localhost:8000", 
                   model: str = "facebook/opt-1.3b") -> bool:
    """Тестирует генерацию текста"""
    print(f"\n🤖 Тестирование генерации с моделью: {model}")
    
    test_prompt = "What is the capital of France?"
    print(f"Вопрос: {test_prompt}")
    
    try:
        response = requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": model,
                "prompt": test_prompt,
                "max_tokens": 50,
                "temperature": 0.7
            },
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        answer = result['choices'][0]['text']
        
        print(f"\n✅ Ответ модели:")
        print(f"   {answer}")
        print(f"\n📊 Статистика:")
        print(f"   - Prompt tokens: {result['usage']['prompt_tokens']}")
        print(f"   - Completion tokens: {result['usage']['completion_tokens']}")
        print(f"   - Total tokens: {result['usage']['total_tokens']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при генерации: {e}")
        return False


def test_openai_compatibility(base_url: str = "http://localhost:8000",
                              model: str = "facebook/opt-1.3b") -> bool:
    """Тестирует совместимость с OpenAI API"""
    print(f"\n🔌 Тестирование OpenAI API совместимости...")
    
    try:
        from openai import OpenAI
        
        client = OpenAI(
            api_key="EMPTY",
            base_url=f"{base_url}/v1"
        )
        
        response = client.completions.create(
            model=model,
            prompt="Hello, how are you?",
            max_tokens=20
        )
        
        print("✅ OpenAI библиотека работает корректно")
        print(f"   Ответ: {response.choices[0].text.strip()}")
        
        return True
        
    except ImportError:
        print("⚠️  OpenAI библиотека не установлена")
        print("   Установите: pip install openai>=1.0.0")
        return False
    except Exception as e:
        print(f"❌ Ошибка OpenAI API: {e}")
        return False


def run_full_test(base_url: str = "http://localhost:8000",
                 model: str = None):
    """Запускает полный набор тестов"""
    print("="*70)
    print("🧪 vLLM Server Test Suite")
    print("="*70)
    
    # 1. Проверка доступности
    if not test_server_health(base_url):
        print("\n❌ Сервер недоступен. Прекращение тестирования.")
        sys.exit(1)
    
    # 2. Получение списка моделей
    models = get_available_models(base_url)
    
    # Если модель не указана, берём первую доступную
    if model is None and models.get('data'):
        model = models['data'][0]['id']
        print(f"\n💡 Используется модель: {model}")
    
    if not model:
        print("\n❌ Не удалось определить модель для тестирования")
        sys.exit(1)
    
    # 3. Тест генерации
    completion_ok = test_completion(base_url, model)
    
    # 4. Тест OpenAI совместимости
    openai_ok = test_openai_compatibility(base_url, model)
    
    # Итоги
    print("\n" + "="*70)
    print("📊 Результаты тестирования:")
    print("="*70)
    print(f"   Доступность сервера: ✅")
    print(f"   Генерация текста: {'✅' if completion_ok else '❌'}")
    print(f"   OpenAI совместимость: {'✅' if openai_ok else '⚠️'}")
    print("="*70)
    
    if completion_ok:
        print("\n✅ Все основные тесты пройдены успешно!")
        print("   Сервер готов к использованию.")
    else:
        print("\n⚠️  Некоторые тесты не прошли.")
        print("   Проверьте логи сервера для подробностей.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Тестирование vLLM сервера"
    )
    
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="Базовый URL vLLM сервера (по умолчанию: http://localhost:8000)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Имя модели для тестирования (по умолчанию: автоопределение)"
    )
    
    args = parser.parse_args()
    
    run_full_test(args.base_url, args.model)
