# ДЗ-14: RAG-система с поиском по собственной базе документов (Milvus + OpenAI)

Векторный поиск для **RAG** на векторной БД **Milvus** (в Docker). Эмбеддинги — **OpenAI**
`text-embedding-3-small`. Реализация — Jupyter-ноутбук [`hw14.ipynb`](hw14.ipynb).

## Почему Milvus
Из рассмотренных вариантов (Pinecone / Chroma / Milvus / ClickHouse) только Milvus даёт в одном
месте всё, что нужно для задания: self-host в Docker, проектирование схемы, тонкая настройка
индексов и сравнение нескольких ANN-алгоритмов.

> ⚠️ **ANNOY в Milvus.** Начиная с Milvus 2.3.0 индекс ANNOY **удалён**
> ([issue #30608](https://github.com/milvus-io/milvus/issues/30608)). В Milvus 2.5 нативно есть
> **IVF** (IVF_FLAT/IVF_SQ8/IVF_PQ), **HNSW**, SCANN, DISKANN. Поэтому **IVF vs HNSW** сравниваем
> прямо в Milvus, а **ANNOY** — через оригинальную библиотеку `annoy` (Spotify) отдельным блоком.

## Что покрыто (по пунктам задания)
**Часть 1 — настройка и индексация**
- ✅ векторная БД Milvus поднята в Docker (`docker-compose.yml`);
- ✅ собственный датасет с метаданными (`documents.json`);
- ✅ эмбеддинги через OpenAI + индексация;
- ✅ оптимальная схема коллекции (векторное поле + скалярные для фильтров);

**Изучение ANN-алгоритмов**
- ✅ сравнение **IVF vs HNSW vs ANNOY** на recall@k и latency;
- ✅ разобраны параметры (`nlist`/`nprobe`, `M`/`efConstruction`/`ef`, `n_trees`/`search_k`);
- ✅ trade-off скорость ↔ точность (сводная таблица + выводы);

**Часть 2 — реализация поиска**
- ✅ семантический поиск по векторам;
- ✅ метрики похожести (cosine / L2 / IP);
- ✅ фильтрация по метаданным (булевы выражения Milvus);
- ✅ подбор `top-k` и search-параметров;
- ✅ бонус: мини-RAG (retrieval + ответ LLM по контексту).

## Запуск

```bash
# 1. Поднять Milvus (etcd + minio + milvus-standalone)
docker compose up -d
docker compose ps                 # дождись статуса healthy у milvus-standalone (~30-60 сек)

# 2. Python-окружение и ключ OpenAI
pip install -r requirements.txt
copy .env.example .env            # Windows;  Linux/macOS: cp .env.example .env
#  -> впиши OPENAI_API_KEY в .env

# 3. Открыть ноутбук
jupyter notebook hw14.ipynb       # либо открыть в VS Code и выполнить ячейки сверху вниз
```

Остановить и (опционально) удалить данные:
```bash
docker compose down               # остановить
docker compose down -v            # + удалить тома (volumes/) с данными Milvus
```

## Что такое эмбеддинги (кратко)
Эмбеддинг — представление текста плотным вектором фиксированной длины; семантически близкие
тексты дают близкие векторы (по косинусу). Варианты:
- **OpenAI**: `text-embedding-3-small` (dim 1536, дёшево — используем его),
  `text-embedding-3-large` (dim 3072, точнее), `text-embedding-ada-002` (легаси).
- **Локальные** (через `sentence-transformers`, бесплатно): `all-MiniLM-L6-v2` (dim 384, быстрый),
  `BAAI/bge-m3`, `intfloat/e5-large` (топ-качество, мультиязычность).

## Файлы
- `docker-compose.yml` — Milvus standalone (3 контейнера) с комментариями.
- `hw14.ipynb` — основной ноутбук (всё с русскими комментариями).
- `documents.json` — датасет: 20 документов с полями `category` / `source` / `year`.
- `requirements.txt`, `.env.example`.
- `volumes/` — данные Milvus (создаётся Docker'ом, в git не попадает).

## Порты
| Порт | Сервис |
|---|---|
| 19530 | Milvus gRPC API (с ним работает `pymilvus`) |
| 9091 | Milvus health/metrics |
| 9001 | веб-консоль MinIO (`minioadmin` / `minioadmin`) |
