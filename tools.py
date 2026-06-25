# -*- coding: utf-8 -*-
"""
LangChain-инструменты (@tool) для чат-ассистента.

Что такое "tool" для LLM:
    Инструмент — это обычная Python-функция с описанием (docstring) и типизированными
    аргументами. LangChain по декоратору @tool превращает её в объект Tool, у которого есть
    имя, описание и JSON-схема входа. Агент (LLM) читает эти описания и сам решает, какой
    инструмент вызвать и с какими аргументами, чтобы ответить на вопрос. Так модель получает
    доступ к актуальным данным и вычислениям, которых нет в её весах.

Здесь три инструмента:
    • web_search  — свежая информация из интернета (DuckDuckGo, без API-ключа);
    • fact_check  — проверка фактов по Wikipedia (REST API, без ключа);
    • calculator  — безопасные арифметические вычисления (numexpr).

Файл самодостаточный: можно импортировать (`from tools import TOOLS`) или запустить напрямую
`python tools.py` для быстрой проверки без LLM.
"""
from __future__ import annotations

import requests
from langchain_core.tools import tool


@tool
def web_search(query: str, max_results: int = 3) -> str:
    """Ищет актуальную информацию в интернете через DuckDuckGo.
    Используй для свежих новостей, текущих событий, цен, дат — всего, чего может не быть
    в памяти модели. Аргумент query — поисковый запрос на естественном языке."""
    # Пакет переименован: новое имя — ddgs, старое — duckduckgo_search. Поддержим оба.
    try:
        from ddgs import DDGS
    except ImportError:  # старая версия
        from duckduckgo_search import DDGS

    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return f"Ошибка поиска: {e}"

    if not hits:
        return "Ничего не найдено."
    return "\n\n".join(
        f"{h.get('title', '')}\n{h.get('body', '')}\nИсточник: {h.get('href', '')}"
        for h in hits
    )


@tool
def fact_check(topic: str, lang: str = "ru") -> str:
    """Проверяет факт/находит справку по теме в Wikipedia и возвращает краткое резюме.
    Используй, когда нужен достоверный энциклопедический факт (кто такой, что это, когда было).
    topic — название статьи или тема; lang — язык вики ('ru' или 'en')."""
    base = f"https://{lang}.wikipedia.org"
    try:
        # 1) находим точное название статьи через opensearch
        r = requests.get(
            f"{base}/w/api.php",
            params={"action": "opensearch", "search": topic, "limit": 1, "format": "json"},
            headers={"User-Agent": "hw15-fact-checker/1.0"},
            timeout=10,
        )
        r.raise_for_status()
        titles = r.json()[1]
        if not titles:
            # запасной вариант — английская вики
            if lang != "en":
                return fact_check.func(topic, lang="en")
            return f"В Wikipedia не найдено статьи по теме «{topic}»."
        title = titles[0]

        # 2) берём краткое резюме статьи через REST summary API
        s = requests.get(
            f"{base}/api/rest_v1/page/summary/{requests.utils.quote(title)}",
            headers={"User-Agent": "hw15-fact-checker/1.0"},
            timeout=10,
        )
        s.raise_for_status()
        data = s.json()
        extract = data.get("extract", "Нет описания.")
        url = data.get("content_urls", {}).get("desktop", {}).get("page", base)
        return f"{title}: {extract}\nИсточник: {url}"
    except Exception as e:
        return f"Ошибка обращения к Wikipedia: {e}"


@tool
def calculator(expression: str) -> str:
    """Вычисляет арифметическое/математическое выражение и возвращает результат.
    Используй для любой математики вместо счёта в уме. Поддерживает + - * / ** , скобки и
    функции sqrt, sin, cos, log, exp. Пример: '23*17 + sqrt(144)'."""
    import numexpr  # безопаснее eval: не исполняет произвольный код
    try:
        result = numexpr.evaluate(expression).item()
        return str(result)
    except Exception as e:
        return f"Не удалось вычислить «{expression}»: {e}"


# Список для импорта в агента
TOOLS = [web_search, fact_check, calculator]


if __name__ == "__main__":
    # Быстрая проверка инструментов без LLM (.invoke вызывает tool с dict-аргументами)
    print("calculator:", calculator.invoke({"expression": "23*17 + sqrt(144)"}))
    print("\nfact_check:", fact_check.invoke({"topic": "Лев Толстой", "lang": "ru"})[:200], "...")
    print("\nweb_search:", web_search.invoke({"query": "погода в Москве сегодня"})[:200], "...")
