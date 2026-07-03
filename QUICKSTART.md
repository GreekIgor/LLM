# 🚀 Быстрый старт - HW9

## 🎯 Самый быстрый способ (Рекомендуется!) — Docker Compose

Стек состоит из двух частей:
- **vLLM** — запускается нативно на хосте (ему нужен доступ к GPU/железу);
- **MLflow + inference-сервис + Prometheus** — поднимаются в Docker Compose.

### Шаг 1. Запустите vLLM на хосте

**Windows (PowerShell):**
```powershell
.\start_vllm.ps1
```

**Linux/Mac:**
```bash
bash start_vllm.sh
```

Скрипт создаст `.venv`, поставит зависимости и запустит vLLM на `http://localhost:8000`.
Дождитесь сообщения `Uvicorn running on http://0.0.0.0:8000` (первая загрузка модели — 5-15 минут).

Модель/устройство можно переопределить переменными окружения:
```bash
MODEL=facebook/opt-125m DEVICE=cpu bash start_vllm.sh
```

### Шаг 2. Поднимите остальной стек в Docker

**В отдельном терминале:**
```bash
docker compose up --build
```

Compose запустит MLflow, inference-сервис и Prometheus, дождавшись healthcheck'ов.
Inference-контейнер обращается к vLLM на хосте через `host.docker.internal:8000`.

После запуска доступны:
- 📊 MLflow UI: http://localhost:5000
- 🤖 vLLM API: http://localhost:8000 (нативно на хосте)
- 🌐 Inference Service: http://localhost:8080
- 📖 API Docs: http://localhost:8080/docs
- 📈 Prometheus: http://localhost:9090 (метрики inference-сервиса)

**Остановка:** `Ctrl+C` в терминале с compose (или `docker compose down`), затем `Ctrl+C` в терминале с vLLM.

---

## 📋 Ручной запуск (если нужен контроль)

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
├── docker-compose.yml         # MLflow + inference + Prometheus
├── Dockerfile                 # Образ inference-сервиса
├── mlflow.Dockerfile          # Образ MLflow tracking server
├── requirements_inference.txt # Зависимости для контейнера inference
├── monitoring/prometheus.yml  # Конфиг Prometheus
├── start_vllm.ps1 / .sh       # Запуск vLLM на хосте
├── start_vllm_server.py       # Лаунчер vLLM (используется скриптами)
├── inference_service.py       # FastAPI inference-сервис
├── test_vllm_server.py        # Скрипт тестирования
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
