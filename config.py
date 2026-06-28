"""
Конфигурация LLM и Ragas-обёрток для ДЗ-19.

Здесь собрана вся работа с провайдером, чтобы остальной код (RAG-приложение и
скрипт оценки) не знал деталей. Используется **OpenRouter** — OpenAI-совместимый
эндпоинт с бесплатными моделями (у автора нет рабочего доступа к OpenAI/Groq).

Что отдаёт модуль:
  * get_chat_client()   -> (openai.OpenAI, model_name)  — для генерации ответов в RAG.
  * get_ragas_llm()     -> LangchainLLMWrapper           — LLM-судья для метрик Ragas.
  * get_ragas_embeddings() -> LangchainEmbeddingsWrapper — эмбеддинги для Answer Relevance.

Все параметры берутся из переменных окружения (.env), см. .env.example.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# Базовый OpenAI-совместимый эндпоинт OpenRouter.
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Модель-генератор (отвечает на вопросы в RAG) и модель-судья (оценивает метрики).
# Их можно развести: судье полезна модель посильнее. По умолчанию — одна и та же.
# ВАЖНО: бесплатные `:free`-модели часто отдают 429 и быстро устаревают.
# Для стабильного прогона в CI лучше задать недорогую платную модель.
GEN_MODEL = os.getenv("RAGAS_GEN_MODEL", "openai/gpt-4o-mini")
JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", GEN_MODEL)

# Лёгкая ONNX-модель эмбеддингов (fastembed) — не тянет torch, работает в CI и на Windows.
EMBED_MODEL = os.getenv("RAGAS_EMBED_MODEL", "BAAI/bge-small-en-v1.5")


def _api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "Не задан OPENROUTER_API_KEY. Скопируй .env.example -> .env и впиши ключ "
            "(https://openrouter.ai/keys), либо экспортируй переменную окружения."
        )
    return key


def get_chat_client():
    """Клиент OpenRouter для генерации ответов + имя модели-генератора."""
    from openai import OpenAI

    client = OpenAI(api_key=_api_key(), base_url=OPENROUTER_BASE_URL)
    return client, GEN_MODEL


def get_ragas_llm():
    """LLM-судья для метрик Ragas (Faithfulness, Answer Relevance, Context Recall)."""
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    chat = ChatOpenAI(
        model=JUDGE_MODEL,
        api_key=_api_key(),
        base_url=OPENROUTER_BASE_URL,
        temperature=0.0,  # детерминизм судьи — важно для повторяемости метрик
        timeout=120,
        max_retries=3,
    )
    return LangchainLLMWrapper(chat)


def get_ragas_embeddings():
    """Эмбеддинги для метрики Answer Relevance (косинус между вопросом и его реконструкцией)."""
    from langchain_community.embeddings import FastEmbedEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(FastEmbedEmbeddings(model_name=EMBED_MODEL))
