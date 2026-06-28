# ДЗ-12: LLM-приложение — суммаризатор документов с мониторингом Langfuse

Суммаризатор документов на **OpenRouter** (OpenAI-совместимый API) с полным циклом наблюдаемости через **Langfuse**.
Реализация — Jupyter notebook [`hw12.ipynb`](hw12.ipynb).

## Что делает
- Загружает документ (`.txt` / `.md` / `.pdf`).
- Режет длинный текст на чанки по токенам (`tiktoken`) с перекрытием.
- Суммаризирует по стратегии **map-reduce**: каждый чанк → промежуточное резюме (map),
  затем объединение в финальное структурированное резюме (reduce).
- Логирует весь пайплайн в Langfuse: промпты, ответы, токены, стоимость, латентность,
  иерархию шагов и оценку качества (score).

## Мониторинг (Langfuse)
Демонстрируются все ключевые сущности:

| Сущность | Реализация |
|---|---|
| **Trace** | один вызов `summarize_document()` |
| **Span** | функции под `@observe()` (`summarize_chunk`, `reduce_summaries`) |
| **Generation** | авто-логирование вызовов LLM через `from langfuse.openai import OpenAI` (клиент направлен на OpenRouter) |
| **Event** | `create_event`: `document-loaded`, `map-stage-complete`, `llm-call-failed` (ERROR) |
| **Score** | `create_score`: `manual-quality` + авто-метрика `compression-ratio` |

Теги/метаданные трейса задаются через `propagate_attributes(tags=..., metadata=...)`.

### Dataset и Experiment
- **Dataset** `hw12-summarization-eval` — тестовый набор пар «вход → ожидаемые ключевые темы»
  (`create_dataset` / `create_dataset_item`).
- **Experiment** `summarizer-v1` — прогон `task` по датасету (`dataset.run_experiment`) с двумя
  кастомными evaluator'ами: `keyword-coverage` (качество) и `conciseness` (краткость).
  В markdown показано, как подключить готовый evaluator через `autoevals` (LLM-as-a-judge).

> Совместимо с Langfuse SDK v3/v4 (протестировано на `langfuse 4.11`).

## Запуск
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
copy .env.example .env            # затем впиши ключи в .env
jupyter notebook hw12.ipynb       # либо открыть в VS Code
```

## Ключи (`.env`)
| Переменная | Откуда взять |
|---|---|
| `OPENROUTER_API_KEY` | openrouter.ai/keys (есть бесплатные модели) |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Langfuse → Project Settings |
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` (EU) или `https://us.cloud.langfuse.com` (US) |

Если `.env` нет — ноутбук запросит ключи интерактивно. Файл `.env` в git не попадает (см. `.gitignore`).

## Файлы
- `hw12.ipynb` — основной ноутбук.
- `sample_document.txt` — пример документа для суммаризации.
- `requirements.txt`, `.env.example`, `.gitignore`.
