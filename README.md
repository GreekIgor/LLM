# ДЗ: Оценка качества QA-генерации с помощью метрик (Трек A)

Сравнение трёх бесплатных LLM на задаче Question Answering (русский язык) по метрикам
качества: **Exact Match, token-F1, BLEU, Semantic Similarity**, а также время генерации и
длина ответа.

## Что внутри

| Файл | Назначение |
|------|------------|
| `qa_metrics_eval.ipynb` | Основной ноутбук: загрузка данных, прогон моделей, метрики, эксперименты, графики |
| `metrics.py` | Реализация метрик (EM, F1, BLEU, Semantic Similarity) + bootstrap-тесты значимости |
| `qa_dataset.jsonl` | Срез датасета (80 примеров), генерируется ноутбуком — фиксирован для воспроизводимости |
| `results.csv` / `summary.csv` | Сырые предсказания и итоговая сводка по метрикам (создаются прогоном) |
| `report.md` | Аналитический отчёт (→ экспортировать в PDF для сдачи) |

## Трек и обоснование

**Трек A — Question Answering.** QA даёт прямой сигнал о фактической точности модели:
есть короткий эталонный ответ, и метрики измеряют именно его воспроизведение, а не
правдоподобность текста. Это удобно для интерпретации ошибок и для решения «какую модель
брать под фактологические задачи».

## Данные

- **Источник:** [`sberbank-ai/sberquad`](https://huggingface.co/datasets/sberbank-ai/sberquad)
  (Hugging Face) — русская версия SQuAD: контекст + вопрос + эталонный короткий ответ.
- **Срез:** 80 примеров из валидационного сплита (seed=42), сохраняются в `qa_dataset.jsonl`.

## Модели (OpenRouter, free)

- `meta-llama/llama-3.3-70b-instruct:free`
- `google/gemini-2.0-flash-exp:free`
- `qwen/qwen-2.5-72b-instruct:free`

## Метрики

**Обязательные:** Semantic Similarity (sentence-transformers, мультиязычная MiniLM),
время генерации, длина ответа.
**Трек A:** token-level F1, Exact Match (с нормализацией текста), BLEU (sacrebleu).

## Эксперименты

1. Сравнение 3 моделей на базовом промпте.
2. Влияние длины/формата промпта: `short` vs `detailed` (A/B).
3. Zero-shot vs few-shot промптинг.
4. Влияние температуры генерации (0.0 / 0.3 / 0.7).
5. Анализ типичных ошибок (лучшие/худшие примеры, расхождение формы и смысла).
6. **Бонус:** bootstrap-доверительные интервалы и парный тест значимости.

## Воспроизведение

> ⚠️ Рекомендуется **Google Colab**: локально на машине автора инференс
> `sentence-transformers`/`transformers` падает с segmentation fault.

1. Получить ключ на [openrouter.ai](https://openrouter.ai/keys).
2. Прописать ключ:
   - локально — в `.env`: `OPENROUTER_API_KEY=sk-or-...`
   - в Colab — через **Secrets** (`OPENROUTER_API_KEY`) или `os.environ`.
3. Положить рядом с ноутбуком `metrics.py` (и `qa_dataset.jsonl`, если уже сгенерирован).
4. Открыть `qa_metrics_eval.ipynb` и выполнить ячейки сверху вниз.
   Ячейка установки зависимостей в начале ноутбука ставит всё необходимое.

Зависимости: `datasets`, `sentence-transformers`, `sacrebleu`, `openai`, `pandas`,
`matplotlib`, `scipy`, `python-dotenv`.

## Самопроверка метрик (без сети/GPU)

```bash
python metrics.py   # прогоняет встроенные assert-ы
```
