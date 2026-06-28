"""
Юнит-тесты RAG-приложения (ДЗ-19).

Проверяют только ретривер и сборку промпта — без обращения к LLM/сети,
поэтому гоняются в CI быстро и без API-ключа.
"""
import json
from pathlib import Path

import pytest

from rag_app import Retriever, build_prompt

GOLDENS = json.loads(
    (Path(__file__).parent / "goldens.json").read_text(encoding="utf-8")
)["items"]


@pytest.fixture(scope="module")
def retriever():
    return Retriever()


def test_retriever_returns_results(retriever):
    docs = retriever.retrieve("Что такое GIL?", top_k=3)
    assert docs, "ретривер должен вернуть хотя бы один документ"
    assert all("text" in d and "score" in d for d in docs)


def test_retriever_respects_top_k(retriever):
    docs = retriever.retrieve("Кто создал Python?", top_k=2)
    assert len(docs) <= 2


def test_retriever_scores_sorted_desc(retriever):
    docs = retriever.retrieve("виртуальное окружение venv", top_k=3)
    scores = [d["score"] for d in docs]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("golden", GOLDENS, ids=[g["id"] for g in GOLDENS])
def test_retriever_finds_expected_context(retriever, golden):
    """Для каждого golden ожидаемый документ должен попадать в top-3 выдачи.

    Это гарантирует, что Context Recall в Ragas не упрётся в плохой ретривер.
    """
    docs = retriever.retrieve(golden["question"], top_k=3)
    found_ids = {d["id"] for d in docs}
    expected = set(golden["expected_context_ids"])
    assert expected & found_ids, (
        f"ожидали один из {expected} в top-3, получили {found_ids}"
    )


def test_build_prompt_contains_question_and_contexts():
    prompt = build_prompt("Вопрос?", ["контекст А", "контекст Б"])
    assert "Вопрос?" in prompt
    assert "контекст А" in prompt and "контекст Б" in prompt
