# ДЗ-19. CI/CD-пайплайн с автотестами и проверкой на галлюцинации (Ragas)

Цель: протестировать LLM-приложение с помощью **Ragas** и встроить проверку качества
в **CI/CD** с quality gates (пайплайн «падает», если метрики ниже порога).

## Что внутри

Простой **RAG QA-бот** по базе знаний о языке Python:

1. **Retriever** ([rag_app.py](rag_app.py)) — TF-IDF + косинусная близость по корпусу
   из 14 документов ([corpus.py](corpus.py)). Детерминированный, CPU, без API-ключа —
   поэтому юнит-тестируется в CI офлайн.
2. **Generator** — LLM через **OpenRouter** (OpenAI-совместимый API) отвечает на вопрос
   *строго по найденному контексту*; промпт запрещает выдумывать факты — это и проверяет
   метрика Faithfulness.

```
вопрос ──▶ Retriever (TF-IDF, top-k=3) ──▶ контекст ──▶ LLM (OpenRouter) ──▶ ответ
                                                              │
                                          Ragas (LLM-судья + эмбеддинги) ──▶ метрики
```

## Структура

| Файл | Назначение |
|------|-----------|
| [corpus.py](corpus.py) | База знаний (14 документов о Python) |
| [rag_app.py](rag_app.py) | RAG: `Retriever` + `generate_answer()`; офлайн self-test |
| [config.py](config.py) | OpenRouter-клиент и обёртки LLM/эмбеддингов для Ragas |
| [tests/goldens.json](tests/goldens.json) | 15 золотых примеров (вопрос + эталонный ответ) |
| [ragas_eval.py](ragas_eval.py) | Прогон Ragas → `ragas_results.json` + `ragas_report.html` |
| [thresholds.json](thresholds.json) | Пороги quality gate |
| [tests/test_rag_app.py](tests/test_rag_app.py) | Юнит-тесты ретривера (офлайн) |
| [tests/test_ragas_quality.py](tests/test_ragas_quality.py) | Quality-gate тесты по метрикам |
| [.github/workflows/ragas-ci.yml](.github/workflows/ragas-ci.yml) | GitHub Actions с quality gates |
| `ragas_results.json` / `ragas_report.html` | Отчёт (JSON + HTML), артефакт прогона |

## Метрики Ragas

| Метрика | Что измеряет | Что нужно для расчёта |
|---------|--------------|------------------------|
| **Faithfulness** | Доля утверждений ответа, подтверждаемых контекстом — **проверка на галлюцинации** | LLM-судья |
| **Answer Relevance** (`answer_relevancy`) | Насколько ответ релевантен вопросу (через реконструкцию вопроса из ответа + косинус эмбеддингов) | LLM + эмбеддинги |
| **Context Recall** (`LLMContextRecall`) | Насколько извлечённый контекст покрывает факты эталонного ответа (`reference`) | LLM-судья + `reference` |

Все метрики ∈ [0, 1], выше — лучше.

## Пороговые значения (quality gates)

Заданы в [thresholds.json](thresholds.json):

| Метрика | Порог |
|---------|-------|
| Faithfulness | **≥ 0.70** |
| Answer Relevance | **≥ 0.70** |
| Context Recall | **≥ 0.70** |

Порог **0.70** — разумный стартовый «потолок терпимости»: ответ, опирающийся на контекст
менее чем на 70%, считаем подозрительным на галлюцинации. Пороги стоит поднимать по мере
стабилизации приложения.

### Текущие результаты (реальный прогон, `ragas_results.json`, 15 примеров)

Судья — `openai/gpt-4o-mini` через OpenRouter, эмбеддинги — `BAAI/bge-small-en-v1.5`:

| Метрика | Среднее | Порог | Статус |
|---------|---------|-------|--------|
| Faithfulness | **1.000** | 0.70 | ✅ PASS |
| Answer Relevance | **0.759** | 0.70 | ✅ PASS |
| Context Recall | **1.000** | 0.70 | ✅ PASS |

Faithfulness = 1.0 на всех примерах — бот не галлюцинирует, отвечает строго по контексту.
Context Recall = 1.0 — TF-IDF-ретривер находит нужный документ для каждого вопроса.
Answer Relevance ниже (0.759) — метрика чувствительна к «многословности»/формулировке ответа
(самый низкий пример — про GIL, 0.58), но в среднем порог пройден.

> `ragas_results.json` пересчитывается командой `python ragas_eval.py` (нужен `OPENROUTER_API_KEY`).
> Числа от прогона к прогону немного плавают — судья недетерминирован даже при temperature=0.

## Запуск локально

```bash
pip install -r requirements.txt
cp .env.example .env          # впиши OPENROUTER_API_KEY (https://openrouter.ai/keys)

python rag_app.py             # офлайн self-test ретривера (без LLM)
python ragas_eval.py          # полный прогон Ragas -> ragas_results.json + ragas_report.html
python ragas_eval.py --limit 3  # быстрый прогон по 3 примерам

pytest -v                     # юнит-тесты + quality gate по артефакту
```

Модели настраиваются переменными окружения (см. [config.py](config.py)):
`RAGAS_GEN_MODEL`, `RAGAS_JUDGE_MODEL`, `RAGAS_EMBED_MODEL`. По умолчанию генератор и судья —
`openai/gpt-4o-mini` (дешёвая платная модель, надёжный structured-output для судьи Ragas;
бесплатные `:free`-модели быстро упираются в `429`), эмбеддинги — `BAAI/bge-small-en-v1.5`
через **fastembed** (ONNX, без torch — не падает на Windows и в CI).

## Автотесты (pytest)

- **`test_rag_app.py`** — офлайн: ретривер возвращает результаты, уважает `top_k`,
  сортирует по убыванию score, и для каждого golden находит ожидаемый документ в top-3.
- **`test_ragas_quality.py`** — quality gates: средние метрики ≥ порогов; не более 20%
  примеров с просадкой Faithfulness (защита от галлюцинаций); согласованность флага
  `gates_passed`. По умолчанию проверяет **закоммиченный** `ragas_results.json`
  (детерминированно, без ключа). С `RAGAS_LIVE=1` + `OPENROUTER_API_KEY` — сначала
  перегенерирует артефакт живым прогоном.

## CI/CD (GitHub Actions)

[.github/workflows/ragas-ci.yml](.github/workflows/ragas-ci.yml) — два job-а:

1. **`unit-and-gate`** (всегда): ставит зависимости, гоняет `pytest -v`. Падает, если
   средние метрики в `ragas_results.json` ниже порогов → **quality gate**. Публикует
   `ragas_results.json` и `ragas_report.html` как артефакты сборки.
2. **`live-eval`** (по `workflow_dispatch`, если задан секрет `OPENROUTER_API_KEY`):
   запускает `python ragas_eval.py` — реальный прогон Ragas через OpenRouter. Скрипт
   возвращает ненулевой код выхода при просадке метрик → job падает. Публикует свежий отчёт.

**Quality gate** реализован в двух местах: в `ragas_eval.py` (код возврата `1` при провале)
и в pytest (`assert value >= threshold`). Любого из них достаточно, чтобы «уронить» пайплайн.

### Как это используется в CI/CD

1. На каждый push/PR гоняются быстрые офлайн-тесты + проверка порогов по артефакту.
2. Отчёты (`json`/`html`) сохраняются как **artifacts** сборки — их можно скачать и посмотреть.
3. Просадка любой метрики ниже порога ⇒ красный билд ⇒ PR не мерджится.
4. Полный (платный, сетевой) прогон Ragas вынесен в ручной/секрет-зависимый job, чтобы
   обычные сборки были детерминированными и не требовали ключей.

### GitLab CI (если вместо GitHub Actions)

Аналогично, `.gitlab-ci.yml`:

```yaml
stages: [test]
ragas-gate:
  stage: test
  image: python:3.12
  script:
    - pip install -r requirements.txt
    - pytest -v               # quality gate по артефакту
  artifacts:
    when: always
    paths: [ragas_results.json, ragas_report.html]
```

## Полезные ссылки

- Ragas — https://github.com/explodinggradients/ragas
- Hugging Face Datasets — https://huggingface.co/docs/datasets
- GitHub Actions — https://docs.github.com/actions
- GitLab CI/CD — https://docs.gitlab.com/ee/ci/
