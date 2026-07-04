# 🚀 Быстрый старт - HW9

## 🎯 Самый быстрый способ (Рекомендуется!) — Docker Compose

Весь стек (MLflow + inference-сервис + Prometheus) поднимается одной командой.
Бэкенд инференса — **OpenRouter** (OpenAI-совместимый API) как замена локального vLLM.

> **Почему не локальный vLLM?** vLLM требует NVIDIA GPU с compute capability ≥ 7.5.
> На доступном железе (Quadro P2000, SM 6.1) это невозможно, а нативной сборки под
> Windows у vLLM нет. Inference-сервис общается с бэкендом по стандартному
> OpenAI-совместимому `/v1/completions`, поэтому OpenRouter подставляется как drop-in.

### Шаг 1. Проверьте ключ в `.env`

Файл `.env` (не коммитится) должен содержать:
```
OPENROUTER_API_KEY=sk-or-v1-...
```
Ключ автоматически подхватывается docker compose и пробрасывается в inference-сервис.

### Шаг 2. Поднимите стек

```bash
docker compose up --build
```

Compose соберёт образы и запустит сервисы по healthcheck'ам. После запуска доступны:
- 📊 MLflow UI: http://localhost:5000
- 🌐 Inference Service: http://localhost:8080
- 📖 API Docs: http://localhost:8080/docs
- 📈 Prometheus: http://localhost:9090

### Шаг 3. Проверьте генерацию

```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The capital of France is","max_tokens":16,"temperature":0}'
```
В ответе будут `output`, счётчики токенов и `latency_seconds`. Каждый запрос
логируется в MLflow (эксперимент `vllm-inference`) и в метрики Prometheus.

**Сменить модель:** отредактируйте `DEFAULT_MODEL` у сервиса `inference` в
[docker-compose.yml](docker-compose.yml) (по умолчанию `meta-llama/llama-3.2-3b-instruct`)
или передайте `"model"` в теле запроса к `/generate`.

**Остановка:** `Ctrl+C`, затем `docker compose down`.

---

## 📋 Ручной запуск локального vLLM (опционально)

> ⚠️ **Требуется NVIDIA GPU с compute capability ≥ 7.5** (T4, RTX 20xx+, A100 и т.п.)
> и Linux/WSL2. На Quadro P2000 (SM 6.1) и под нативным Windows этот путь не работает —
> используйте Docker Compose с OpenRouter выше. Раздел оставлен для машин с подходящим GPU.

## Шаг 1: Установка зависимостей (5 минут)

```bash
pip install -r requirements_hw9.txt
```

## Шаг 2: Запуск vLLM сервера (2-15 минут на загрузку модели)

### Вариант A: Лёгкая модель для CPU
```bash
python start_vllm_server.py --model facebook/opt-1.3b --device cpu
```

### Вариант B: Мощная модель для GPU
```bash
python start_vllm_server.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

**Дождитесь сообщения:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## Шаг 3: Проверка работы (1 минута)

**В новом терминале:**
```bash
python test_vllm_server.py
```

Ожидаемый результат:
```
✅ Сервер доступен и работает
✅ Доступные модели: facebook/opt-1.3b
✅ Генерация текста работает
✅ OpenAI совместимость OK
```

## Шаг 4: Выполнение задания (30-60 минут)

1. Откройте **hw9_vllm_mlflow.ipynb**
2. Выполните все ячейки по порядку
3. Изучите результаты

## Шаг 5: Просмотр результатов в MLflow (5 минут)

**В новом терминале:**
```bash
mlflow ui
```

Откройте в браузере: http://localhost:5000

---

## 📊 Итоговая структура файлов

```
LLM/
├── hw9_vllm_mlflow.ipynb      # Основной ноутбук с заданием
├── docker-compose.yml         # MLflow + inference + Prometheus (бэкенд — OpenRouter)
├── Dockerfile                 # Образ inference-сервиса
├── mlflow.Dockerfile          # Образ MLflow tracking server
├── requirements_inference.txt # Зависимости для контейнера inference
├── monitoring/prometheus.yml  # Конфиг Prometheus
├── inference_service.py       # FastAPI inference-сервис (OpenAI-совместимый бэкенд)
├── .env                       # Ключи, в т.ч. OPENROUTER_API_KEY (не коммитится)
├── start_vllm_server.py       # Лаунчер локального vLLM (опционально, нужен GPU SM≥7.5)
├── test_vllm_server.py        # Тест локального vLLM-сервера
├── requirements_hw9.txt       # Полные зависимости (vllm, torch, jupyter)
├── HW9_README.md              # Подробная документация
└── QUICKSTART.md              # Этот файл
```

---

## ⚡ Команды одной строкой

### Запустить всё сразу:

**Терминал 1 - vLLM сервер:**
```bash
python start_vllm_server.py --model facebook/opt-1.3b
```

**Терминал 2 - Тест:**
```bash
sleep 60 && python test_vllm_server.py
```

**Терминал 3 - Jupyter:**
```bash
jupyter notebook hw9_vllm_mlflow.ipynb
```

**Терминал 4 - MLflow (после выполнения ноутбука):**
```bash
mlflow ui
```

---

## 🐛 Быстрое решение проблем

### Сервер не запускается
```bash
# Проверьте, свободен ли порт
netstat -ano | findstr :8000

# Попробуйте другой порт
python start_vllm_server.py --port 8001
```

### Out of Memory
```bash
# Используйте самую лёгкую модель
python start_vllm_server.py --model facebook/opt-125m --device cpu
```

### Модель долго грузится
**Это нормально!** Первая загрузка занимает 5-15 минут.

---

## ✅ Готово!

Теперь вы можете:
- ✅ Локально запускать LLM через vLLM
- ✅ Взаимодействовать через OpenAI API
- ✅ Трекать эксперименты в MLflow
- ✅ Использовать LLM-as-a-Judge паттерн

**Если остались вопросы, смотрите [HW9_README.md](HW9_README.md)**
