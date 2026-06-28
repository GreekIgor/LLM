"""
Простое RAG-приложение (QA-бот по базе знаний о Python) — ДЗ-19.

Пайплайн классический:
  1. Retriever — по вопросу ищет top-k релевантных документов в базе знаний
     (TF-IDF + косинусная близость; детерминированно, CPU, без тяжёлых зависимостей).
  2. Generator — LLM (через OpenRouter) отвечает на вопрос СТРОГО по найденному
     контексту. Промпт инструктирует не выдумывать факты — это и проверяет Ragas.

Отдельный ретривер на TF-IDF выбран намеренно: он не требует API-ключа и эмбеддингов,
поэтому юнит-тесты ретривера и golden-проверки гоняются в CI без обращения к сети.

Запуск self-test:  python rag_app.py
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from corpus import get_corpus

# Сколько документов подкладывать в контекст ответа.
DEFAULT_TOP_K = 3

SYSTEM_PROMPT = (
    "Ты — ассистент, отвечающий на вопросы СТРОГО по предоставленному контексту. "
    "Используй только факты из контекста. Если ответа в контексте нет — честно скажи, "
    "что информации недостаточно. Не придумывай детали. Отвечай кратко, на русском языке."
)

USER_PROMPT_TEMPLATE = (
    "Контекст:\n{context}\n\n"
    "Вопрос: {question}\n\n"
    "Ответ (только по контексту):"
)


class Retriever:
    """TF-IDF ретривер по базе знаний. Строит индекс один раз при инициализации."""

    def __init__(self, corpus: list[dict] | None = None):
        self.corpus = corpus if corpus is not None else get_corpus()
        self._texts = [d["text"] for d in self.corpus]
        # ngram_range=(1,2) немного улучшает попадание по словосочетаниям.
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self._matrix = self._vectorizer.fit_transform(self._texts)

    def retrieve(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        """Возвращает top_k документов, отсортированных по убыванию релевантности.

        Каждый элемент: {"id", "text", "score"}.
        """
        q_vec = self._vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self._matrix)[0]
        # argsort по убыванию; берём только документы с ненулевой близостью.
        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )
        results = []
        for i in ranked[:top_k]:
            if scores[i] <= 0.0:
                continue
            doc = dict(self.corpus[i])
            doc["score"] = float(scores[i])
            results.append(doc)
        return results


def build_prompt(question: str, contexts: list[str]) -> str:
    """Собирает пользовательский промпт из вопроса и найденных контекстов."""
    context_block = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    return USER_PROMPT_TEMPLATE.format(context=context_block, question=question)


def generate_answer(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    retriever: Retriever | None = None,
) -> dict:
    """Полный RAG-проход: retrieve -> generate.

    Возвращает словарь:
      {
        "question": ...,
        "answer": ...,                  # ответ LLM
        "contexts": [str, ...],         # тексты подложенных документов
        "context_ids": [str, ...],      # их id (для отладки/анализа)
      }

    Требует OPENROUTER_API_KEY (см. config.py). Сетевая часть вынесена сюда, чтобы
    ретривер можно было тестировать отдельно без обращения к LLM.
    """
    retriever = retriever or Retriever()
    docs = retriever.retrieve(question, top_k=top_k)
    contexts = [d["text"] for d in docs]

    # Ленивый импорт: тесты ретривера не должны требовать openai/ключ.
    from config import get_chat_client

    client, model = get_chat_client()
    completion = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, contexts)},
        ],
    )
    answer = (completion.choices[0].message.content or "").strip()

    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "context_ids": [d["id"] for d in docs],
    }


if __name__ == "__main__":
    # Offline self-test: проверяем только ретривер (без обращения к LLM/сети).
    print("[*] Self-test ретривера (без LLM)\n")
    r = Retriever()
    demo_questions = [
        "Кто создал язык Python и когда?",
        "Что такое GIL?",
        "Когда прекратилась поддержка Python 2?",
        "Зачем нужно виртуальное окружение?",
    ]
    for q in demo_questions:
        docs = r.retrieve(q, top_k=2)
        ids = ", ".join(f"{d['id']}({d['score']:.2f})" for d in docs)
        print(f"Q: {q}\n   -> top docs: {ids}\n")
    print("[OK] Ретривер работает. Для полного RAG-ответа нужен OPENROUTER_API_KEY.")
