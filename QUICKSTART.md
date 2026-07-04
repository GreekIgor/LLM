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

## 📓 Работа с заданием (ноутбук)

1. Поднимите стек: `docker compose up --build` (см. выше).
2. Откройте **hw9_vllm_mlflow.ipynb** и выполните ячейки по порядку.
   Обращения к бэкенду идут через inference-сервис (`http://localhost:8080`) → OpenRouter.
3. Результаты экспериментов смотрите в **MLflow UI**: http://localhost:5000
   (эксперимент `vllm-inference`).

---

## ⛔ Про локальный vLLM (не работает на этой машине)

Скрипты `start_vllm_server.py` / `test_vllm_server.py` запускают **нативный vLLM** и
на данном железе всегда падают:

```
ModuleNotFoundError: No module named 'vllm._C_stable_libtorch'
```

Причина принципиальная, `--device cpu` не помогает:
- у vLLM **нет нативной сборки под Windows** (нет скомпилированного C-расширения);
- vLLM требует NVIDIA GPU с **compute capability ≥ 7.5**, а Quadro P2000 — 6.1 (Pascal);
- свежий Docker-образ vLLM собран под CUDA 13, которую не тянет установленный драйвер.

Поэтому бэкендом служит OpenRouter (см. основной раздел). Локальный vLLM возможен
только на машине с поддерживаемым GPU (T4 / RTX 20xx+ / A100 …) под Linux/WSL2.

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

```bash
# Поднять весь стек (MLflow + inference + Prometheus)
docker compose up --build

# Проверить генерацию
curl -X POST http://localhost:8080/generate -H "Content-Type: application/json" \
  -d '{"prompt":"Hello, world!","max_tokens":32}'

# Открыть ноутбук с заданием
jupyter notebook hw9_vllm_mlflow.ipynb

# Остановить
docker compose down
```

---

## 🐛 Быстрое решение проблем

### `docker compose up` падает на inference
Проверьте, что в `.env` задан `OPENROUTER_API_KEY`. Ключ подставляется в
переменную `VLLM_API_KEY` контейнера через интерполяцию compose.

### `/generate` → 401/403 от бэкенда
Ключ OpenRouter недействителен или исчерпан лимит. Проверьте ключ на
https://openrouter.ai/settings/keys.

### `/generate` → 429 (rate limit)
Выбранная модель перегружена. Смените `DEFAULT_MODEL` в
[docker-compose.yml](docker-compose.yml) на другую платную модель.

### MLflow-run не появляется / 403 «DNS rebinding»
Пересоберите mlflow (`docker compose up -d --build mlflow`) — сервер запускается
с `--allowed-hosts=*`, что разрешает обращения по хосту `mlflow:5000`.

### Порт занят
Занятый порт (5000/8080/9090) освободите или поменяйте маппинг в compose.

---

## ✅ Готово!

Теперь вы можете:
- ✅ Гонять инференс через OpenAI-совместимый бэкенд (OpenRouter)
- ✅ Трекать эксперименты в MLflow (`vllm-inference`)
- ✅ Снимать метрики в Prometheus (`llm_requests_total`, latency и др.)
- ✅ Использовать LLM-as-a-Judge паттерн

**Если остались вопросы, смотрите [HW9_README.md](HW9_README.md)**
