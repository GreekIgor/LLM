# ДЗ-17: Извлечение сущностей и событий из текстов (NER + Information Extraction)

Извлекаем структурированную информацию (сущности и отношения) из коммерческих контрактов
**[CUAD](https://huggingface.co/datasets/theatticusproject/cuad)** с помощью локально
развёрнутых инструктивных LLM в режиме **zero/few-shot** (подход «LLM-as-extractor»):
модель получает текст и возвращает строгий JSON по фиксированной схеме сущностей.

**Схема (7 типов):** `PERSON, ORG, MONEY, DATE, CONTRACT_TYPE, OBLIGATION, JURISDICTION`.

## Структура

| Файл | Что делает |
|---|---|
| [`ie_extractor.py`](ie_extractor.py) | Ядро: промпт со схемой + one-shot, устойчивый парсер JSON, обёртка `Extractor` (load / `extract` / `extract_batch`) |
| [`data_prep.py`](data_prep.py) | Загрузка CUAD (parquet-ветка хаба), подвыборка, маппинг категорий CUAD → наша схема и сборка **gold** |
| [`evaluate.py`](evaluate.py) | Метрики precision / recall / F1 с нечётким сопоставлением спанов (per-type + micro) |
| [`benchmark.py`](benchmark.py) | Throughput (tokens/sec, docs/sec), RAM/VRAM, сравнение моделей и `batch_size` |
| [`app.py`](app.py) | Demo (Gradio): текст контракта → подсвеченные сущности + JSON |
| [`ie_extraction.ipynb`](ie_extraction.ipynb) | Основной ноутбук: этапы 1–4 локально на маленьких моделях (Qwen2.5 0.5B/1.5B, CPU) |
| [`colab_7b.ipynb`](colab_7b.ipynb) | Colab T4: этап 1 на 7B — Mistral-7B / Llama-2-7B, **4-bit quantized vs full precision**, VRAM |
| `requirements.txt` | зависимости |

## Этапы (по заданию)

1. **Локальное развёртывание.** Локально (CPU) — две модели Qwen2.5 (0.5B и 1.5B), full
   precision. Сравнение 7B *quantized vs full precision* — в `colab_7b.ipynb` (нужен CUDA).
2. **Подготовка данных.** Подвыборка CUAD (100–200 для demo, 500–1K для прогона), промпты со
   строгой JSON-схемой + one-shot пример.
3. **Оптимизация для IE.** Batch processing (`extract_batch`), left-padding, обрезка входа,
   измерение throughput.
4. **Анализ производительности.** Скорость (tokens/sec, docs/sec), качество (precision/recall/F1),
   ресурсы (RAM локально, VRAM на GPU).

## Откуда берётся gold для метрик

CUAD размечен юристами по 41 категории. Несколько **экстрактивных** категорий напрямую
ложатся на нашу схему — из них строим gold без ручной разметки:

| Категория CUAD | Тип сущности |
|---|---|
| Document Name | `CONTRACT_TYPE` |
| Parties | `ORG` |
| Agreement / Effective / Expiration Date | `DATE` |
| Governing Law | `JURISDICTION` |

По `PERSON / MONEY / OBLIGATION` прямого gold в CUAD нет — модель их извлекает (видно в demo),
но в метриках они не учитываются, чтобы не штрафовать несправедливо. Сопоставление спанов —
нечёткое (нормализация регистра/пунктуации + перекрытие токенов), т.к. модель может писать
«Acme Corp.» там, где gold «Acme Corp».

## Запуск

```bash
pip install -r requirements.txt
```

Быстрые offline-самопроверки (без моделей и без сети):

```bash
python ie_extractor.py     # тесты парсера JSON
python evaluate.py         # тесты метрик P/R/F1
python data_prep.py        # загрузит подвыборку CUAD (нужна сеть) и покажет gold
```

Бенчмарк и demo (устройство выбирается автоматически, см. ниже):

```bash
python benchmark.py --models Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct \
                    --n-docs 20 --batch-sizes 1 4       # --device auto: GPU если доступна
python app.py              # Gradio на http://127.0.0.1:7860
```

## Про железо и где запускать inference

Способ inference выбирается через `LLM_PROVIDER` в `.env`.

**1. Локально (`LLM_PROVIDER=local`, по умолчанию).** Доступна NVIDIA **Quadro P2000 (5 ГБ)**;
`Extractor` сам определяет железо (`device="auto"`) — при наличии CUDA грузит модель на GPU в
`float16`, иначе откатывается на CPU/`float32`. Маленькие Qwen2.5 (0.5B/1.5B) в 5 ГБ помещаются
свободно.

```bash
python app.py                                           # авто-GPU
python benchmark.py --device cuda --dtype float16 --batch-sizes 1 4 8
```

**2. Через API (`LLM_PROVIDER=openrouter`).** Крупные модели (7B+/70B), не влезающие в 5 ГБ
видеокарты, гоняем через OpenRouter (OpenAI-совместимый API). Ключ — `OPENROUTER_API_KEY`, модель
— `OPENROUTER_MODEL` в `.env`. Интерфейс тот же (`extract` / `extract_batch`), поэтому `app.py` и
`benchmark.py` работают без изменений. Тот же путь подходит и для LM Studio (`LLM_PROVIDER=lmstudio`).

* **`bitsandbytes` (4-bit квантование) требует CUDA** → сравнение 7B quantized vs full
  вынесено в **Google Colab** (`Runtime → T4 GPU`), ноутбук [`colab_7b.ipynb`](colab_7b.ipynb).
  На Windows `bitsandbytes` капризен, а 7B в полном fp16 (~14 ГБ) в 5 ГБ P2000 не помещается —
  через OpenRouter 7B доступна и без Colab.

## Что протестировано локально

* `ie_extractor.parse_entities` — 6 кейсов (чистый JSON, markdown-блок, болтовня вокруг,
  строка вместо списка, битый JSON → regex-fallback, мусор) — ✅.
* `evaluate` — нечёткий матчинг и арифметика TP/FP/FN — ✅.
* `data_prep` — реальная загрузка CUAD (22 450 строк), сборка gold по ORG/DATE/CONTRACT_TYPE — ✅.
* Импорты всех модулей и логика подсветки `app._to_highlighted` — ✅.
* Локальный inference: `transformers 4.57` + токенайзер Qwen грузятся без ошибок; модель
  выполняется на CPU и на GPU (Quadro P2000, CUDA доступна).
