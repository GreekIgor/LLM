# ДЗ-16: Дообучение модели (LoRA) + интеграция с внешними инструментами

Трек B — продвинутый чат-ассистент. Дообучаем **Qwen2.5-1.5B-Instruct** методом **QLoRA** под
качественные диалоги, затем даём модели доступ к внешним **инструментам** (LangChain `@tool`)
и показываем интеграцию через **ReAct-агента**.

## Структура
| Файл | Что делает |
|---|---|
| [`finetune_lora.ipynb`](finetune_lora.ipynb) | Часть 1: QLoRA fine-tuning на сабсете `lmsys-chat-1m`, сравнение «до/после», сохранение адаптера |
| [`tools.py`](tools.py) | Часть 2: инструменты `@tool` — `web_search`, `fact_check`, `calculator` |
| [`agent_demo.ipynb`](agent_demo.ipynb) | Часть 3: ReAct-агент — дообученная модель вызывает инструменты в реальных сценариях |
| `requirements.txt`, `.env.example` | зависимости и ключи |

## ⚠️ Про железо
Локально NVIDIA GPU нет (только Intel HD 520), а `bitsandbytes` (4-бит QLoRA) требует CUDA.
Поэтому **`finetune_lora.ipynb` рассчитан на Google Colab** (Runtime → Change runtime type → **T4 GPU**)
или другую машину с CUDA. `tools.py` и `agent_demo.ipynb` работают и на CPU (модель 1.5B медленно,
но запускается; на GPU — быстро).

## Запуск

### Часть 1 — fine-tuning (в Colab)
1. Открой `finetune_lora.ipynb` в Colab, включи GPU.
2. Получи доступ к gated-датасету: прими условия на
   [lmsys/lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) и задай `HF_TOKEN`.
   *(Нет доступа — ноутбук сам переключится на открытый `ultrachat_200k`.)*
3. Выполни ячейки сверху вниз: установка → загрузка модели (4-бит) → baseline → датасет →
   LoRA-обучение → сравнение «до/после» → сохранение адаптера.

### Части 2–3 — инструменты и агент
```bash
pip install -r requirements.txt
cp .env.example .env          # Windows: copy .env.example .env
python tools.py               # быстрый тест инструментов без LLM
jupyter notebook agent_demo.ipynb
```
В `agent_demo.ipynb` поставь `LOAD_ADAPTER = True`, если обучил адаптер из части 1
(иначе используется базовая модель — демо инструментов всё равно работает).

## Ключевые понятия
- **LoRA / PEFT** — обучаем не все веса, а маленькие низкоранговые добавки (`r=16`) в слои
  attention/MLP; базовая модель заморожена. Обучается ~доли процента параметров.
- **QLoRA** — LoRA поверх 4-битной (nf4) модели: влезает в один бесплатный GPU.
- **LangChain `@tool`** — функция + описание + схема входа; по описанию агент решает, что вызвать.
- **ReAct** — цикл *Reasoning + Acting*: модель пишет `Thought/Action/Action Input`, система
  выполняет инструмент и возвращает `Observation`, пока не появится `Final Answer`.

## Инструменты (`tools.py`)
| Tool | Назначение | Зависимости |
|---|---|---|
| `web_search` | свежая информация из интернета | DuckDuckGo (`ddgs`), без ключа |
| `fact_check` | факты/справки из Wikipedia | REST API, без ключа |
| `calculator` | безопасные вычисления | `numexpr` |

## Демо-сценарии (`agent_demo.ipynb`)
1. **Математика** → `calculator`.
2. **Проверка факта** («кто написал Войну и мир») → `fact_check`.
3. **Свежие данные** (новости про Milvus 2.5) → `web_search`.
4. **Многоступенчатое рассуждение** (высота Эвереста в метрах → перевод в футы) →
   `fact_check` + `calculator`.

> Качество следования формату ReAct у модели 1.5B ограничено — возможны срывы формата.
> Это ожидаемо для демонстрации; лечится дообучением под tool-use или моделью побольше.
