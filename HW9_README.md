# 🚀 ДЗ №9: Трекинг LLM-инференса с vLLM

## 📚 Описание

Домашнее задание по освоению полного цикла использования LLM в роли оценщика (LLM-as-a-Judge):
- Развёртывание локальной модели через vLLM
- Взаимодействие через OpenAI-совместимый API
- Интеграция с MLflow для трекинга экспериментов
- Создание кастомных метрик оценки качества

---

## 📋 Требования

### Установка зависимостей:

```bash
pip install vllm
pip install openai>=1.0.0
pip install httpx requests
pip install mlflow
pip install pandas numpy matplotlib seaborn
```

### Требования к железу:

**Вариант 1: CPU (минимальные требования)**
- RAM: 8+ GB
- Модель: `facebook/opt-125m` или `facebook/opt-1.3b`

**Вариант 2: GPU (рекомендуется)**
- GPU: NVIDIA с 8+ GB VRAM
- CUDA: 11.8+
- Модель: `meta-llama/Meta-Llama-3-8B-Instruct`

---

## 🎯 Структура задания

### Часть 1: Развёртка vLLM сервера

#### Способ 1: Через Python скрипт

```bash
python start_vllm_server.py --model facebook/opt-1.3b --port 8000
```

#### Способ 2: Напрямую через vLLM

**Для CPU:**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model facebook/opt-1.3b \
    --host 0.0.0.0 \
    --port 8000 \
    --device cpu
```

**Для GPU:**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000
```

#### Проверка работы сервера:

```bash
curl http://localhost:8000/v1/models
```

Ожидаемый ответ:
```json
{
  "object": "list",
  "data": [
    {
      "id": "facebook/opt-1.3b",
      "object": "model",
      ...
    }
  ]
}
```

---

### Часть 2: Выполнение ноутбука

1. Откройте `hw9_vllm_mlflow.ipynb`
2. Убедитесь, что vLLM сервер запущен
3. Выполните ячейки по порядку:
   - Проверка подключения
   - HTTP запросы
   - OpenAI API
   - MLflow эксперименты

---

### Часть 3: Просмотр результатов в MLflow

```bash
mlflow ui
```

Откройте в браузере: `http://localhost:5000`

---

## 📖 Подробные инструкции

### 1. Развёртка vLLM

**Шаг 1:** Установите vLLM
```bash
pip install vllm
```

**Шаг 2:** Запустите сервер (выберите модель под ваше железо)

Для слабых машин:
```bash
python start_vllm_server.py --model facebook/opt-125m --device cpu
```

Для средних машин:
```bash
python start_vllm_server.py --model facebook/opt-1.3b
```

Для мощных машин с GPU:
```bash
python start_vllm_server.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

**Шаг 3:** Дождитесь загрузки модели (может занять несколько минут)

Вы увидите:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 2. Взаимодействие с моделью

Ноутбук содержит два метода:

**Метод 1: HTTP запросы (requests)**
```python
import requests

response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "model": "facebook/opt-1.3b",
        "prompt": "What is the capital of Germany?",
        "max_tokens": 100
    }
)
```

**Метод 2: OpenAI библиотека**
```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1"
)

response = client.completions.create(
    model="facebook/opt-1.3b",
    prompt="What is the capital of Germany?",
    max_tokens=100
)
```

### 3. MLflow интеграция

Ноутбук автоматически:
- Создаёт кастомную метрику `relevance`
- Использует локальную модель как "судью"
- Оценивает тестовый датасет
- Логирует результаты в MLflow

**Просмотр результатов:**
```bash
mlflow ui
```

---

## 🎓 Что вы освоите

### ✅ Технические навыки:
1. Развёртывание LLM через vLLM
2. Работа с OpenAI-совместимым API
3. HTTP запросы к LLM серверам
4. Создание кастомных метрик в MLflow
5. LLM-as-a-Judge паттерн
6. **FastAPI production сервисы**
7. **Prometheus метрики для мониторинга**
8. **Автоматизированное тестирование моделей**

### ✅ Практическое применение:
- Автоматическая оценка качества ответов
- A/B тестирование разных моделей
- Мониторинг качества в продакшене
- Оценка результатов fine-tuning
- **Production-ready деплой с метриками**
- **Continuous benchmarking pipeline**

---

## 🐛 Troubleshooting

### Проблема: "Connection refused" при подключении

**Решение:**
1. Проверьте, запущен ли vLLM сервер
2. Убедитесь, что используется правильный порт (8000)
3. Проверьте логи сервера на ошибки

### Проблема: "Out of memory"

**Решение:**
1. Используйте меньшую модель (`facebook/opt-125m`)
2. Добавьте параметр `--device cpu`
3. Уменьшите `max_tokens` в запросах

### Проблема: Модель долго загружается

**Это нормально!** Первая загрузка может занять 5-15 минут в зависимости от:
- Размера модели
- Скорости интернета
- Скорости диска

### Проблема: MLflow не показывает результаты

**Решение:**
1. Проверьте, что эксперимент запущен: `mlflow.set_experiment("vllm_llm_as_judge")`
2. Убедитесь, что ячейка с логированием выполнена
3. Обновите страницу MLflow UI

---

## 📊 Результаты

После выполнения задания вы получите:

1. **Запущенный vLLM сервер** с локальной моделью
2. **Примеры запросов** через HTTP и OpenAI API
3. **MLflow эксперимент** с метриками оценки качества
4. **Оценённый датасет** с рейтингами от LLM-судьи
5. **FastAPI production сервис** с Prometheus метриками
6. **Автоматический бенчмаркинг** с выбором лучшей модели

---

## 🏗️ Production Components (NEW!)

### inference_service.py - FastAPI Production Service

Полноценный production-ready сервис с:
- ✅ **Prometheus метрики**: requests, tokens, latency, errors, active requests
- ✅ **Health checks**: `/health` endpoint
- ✅ **OpenAPI документация**: автоматически сгенерированная `/docs`
- ✅ **MLflow интеграция**: автоматическое логирование всех запросов
- ✅ **Error handling**: правильная обработка ошибок и таймаутов

**Запуск:**
```bash
python inference_service.py
```

**Доступные эндпоинты:**
- `GET /health` - проверка работоспособности
- `GET /metrics` - Prometheus метрики
- `POST /generate` - генерация текста
- `GET /docs` - Swagger UI

### benchmark_vllm_models.py - Automated Benchmarking

Автоматизированное тестирование моделей:
- ✅ Сравнение нескольких моделей на Q&A задачах
- ✅ Вычисление F1 Score и Exact Match
- ✅ Измерение latency и throughput
- ✅ Автоматический выбор лучшей модели
- ✅ Логирование всех результатов в MLflow

**Запуск:**
```bash
export MODELS="facebook/opt-1.3b,EleutherAI/gpt-neo-125M"
python benchmark_vllm_models.py
```

**Результаты:**
- `benchmark_summary.json` - сводка всех результатов
- `best_model.txt` - имя лучшей модели
- MLflow runs для каждой модели

### start_all_services.sh / .ps1 - One-Command Setup

Автоматический запуск всех компонентов:
- MLflow tracking server
- vLLM server с загрузкой модели
- FastAPI inference service
- Автоматическое тестирование

**Использование:**

Linux/Mac:
```bash
bash start_all_services.sh
```

Windows:
```powershell
.\start_all_services.ps1
```

---

## 📚 Дополнительные материалы

- [vLLM Documentation](https://docs.vllm.ai/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [LLM-as-a-Judge Pattern](https://arxiv.org/abs/2306.05685)

---

## ✅ Чек-лист выполнения

- [ ] vLLM сервер запущен и доступен
- [ ] Успешный HTTP запрос к модели
- [ ] Успешный запрос через OpenAI библиотеку
- [ ] Кастомная метрика создана
- [ ] Эксперимент выполнен и залогирован в MLflow
- [ ] Результаты видны в MLflow UI

---

**Удачи в выполнении задания! 🚀**
