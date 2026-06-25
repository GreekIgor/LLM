# -*- coding: utf-8 -*-
"""
Подготовка данных из CUAD (Contract Understanding Atticus Dataset).

Что такое CUAD:
    510 коммерческих контрактов, размеченных юристами по 41 категории клауз. На HF лежит в
    SQuAD-подобном виде: одна строка = (контракт, вопрос по одной категории, найденные
    ответы-спаны). Колонки: id, title, context, question, answers{text:[...], answer_start:[...]}.

Зачем нам это:
    1) Берём короткие фрагменты контрактов как ВХОД для извлечения сущностей.
    2) Несколько экстрактивных категорий CUAD напрямую ложатся на нашу схему сущностей —
       значит, из человеческой разметки можно построить GOLD-стандарт и честно мерить
       precision/recall (см. CATEGORY_TO_ENTITY ниже).

Запуск напрямую:  python data_prep.py
    → загрузит маленькую подвыборку CUAD (или синтетический фолбэк без сети),
      покажет пример документа + построенный gold и сохранит subset в JSON.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ie_extractor import ENTITY_TYPES

# ─────────────────────────────────────────────────────────────────────────────
# Маппинг экстрактивных категорий CUAD → наша схема сущностей.
# Только категории, где человеческий ответ — это спан нужного типа.
# (MONEY и OBLIGATION в CUAD прямого аналога не имеют — для них gold не строим,
#  но модель всё равно извлекает их для demo.)
# ─────────────────────────────────────────────────────────────────────────────
CATEGORY_TO_ENTITY: dict[str, str] = {
    "Document Name": "CONTRACT_TYPE",
    "Parties": "ORG",
    "Agreement Date": "DATE",
    "Effective Date": "DATE",
    "Expiration Date": "DATE",
    "Governing Law": "JURISDICTION",
}

# Категории, по которым у нас есть gold (для отчёта о покрытии метрик).
GOLD_ENTITY_TYPES: list[str] = sorted(set(CATEGORY_TO_ENTITY.values()))

_CATEGORY_RE = re.compile(r'"([^"]+)"')


def category_from_question(question: str) -> str | None:
    """Достаёт имя категории из вопроса CUAD.

    Вопрос выглядит как:
      'Highlight the parts (if any) of this contract related to "Document Name" that ...'
    Берём первую строку в кавычках."""
    m = _CATEGORY_RE.search(question or "")
    return m.group(1).strip() if m else None


@dataclass
class ContractSample:
    """Один документ для извлечения + его gold-разметка."""

    doc_id: str
    title: str
    text: str
    gold: dict[str, list[str]] = field(default_factory=lambda: {t: [] for t in ENTITY_TYPES})

    def to_dict(self) -> dict[str, Any]:
        return {"doc_id": self.doc_id, "title": self.title, "text": self.text, "gold": self.gold}


# ─────────────────────────────────────────────────────────────────────────────
# Загрузка и сборка подвыборки
# ─────────────────────────────────────────────────────────────────────────────
def _add_gold(gold: dict[str, list[str]], entity: str, values: list[str]) -> None:
    seen = set(gold[entity])
    for v in values:
        v = (v or "").strip()
        if v and v not in seen:
            seen.add(v)
            gold[entity].append(v)


def load_cuad_subset(
    n_docs: int = 100,
    max_chars: int = 3000,
    split: str = "train",
    seed: int = 42,
) -> list[ContractSample]:
    """Грузит CUAD с HuggingFace, группирует строки по контракту и строит gold.

    n_docs    — сколько контрактов взять (100-200 для demo, 500-1K для полноценного прогона);
    max_chars — обрезка текста контракта (CUAD-контракты огромные; для CPU берём начало,
                где обычно и находятся стороны/дата/тип — преамбула договора).

    При отсутствии сети/датасета бросает исключение — вызывающий код может перейти на
    synthetic_subset()."""
    from datasets import load_dataset

    # SQuAD-формат (train/test) лежит в *-qa репозитории. В новых версиях datasets (>=3)
    # загрузочные скрипты отключены, поэтому грузим авто-сконвертированный parquet
    # из служебной ветки refs/convert/parquet (HF делает её автоматически).
    last_err: Exception | None = None
    ds = None
    attempts = [
        dict(path="theatticusproject/cuad-qa", split=split, revision="refs/convert/parquet"),
        dict(path="theatticusproject/cuad-qa", split=split),  # на старых datasets со скриптом
    ]
    for kw in attempts:
        try:
            ds = load_dataset(**kw)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
    if ds is None:
        raise RuntimeError(f"Не удалось загрузить CUAD: {last_err}")

    # группируем строки по контракту (title), собирая gold из ответов нужных категорий
    by_contract: dict[str, ContractSample] = {}
    order: list[str] = []
    for row in ds:
        title = row.get("title") or row.get("id") or "unknown"
        if title not in by_contract:
            context = (row.get("context") or "")[:max_chars]
            by_contract[title] = ContractSample(
                doc_id=str(row.get("id", title)),
                title=str(title),
                text=context,
                gold={t: [] for t in ENTITY_TYPES},
            )
            order.append(title)

        category = category_from_question(row.get("question", ""))
        entity = CATEGORY_TO_ENTITY.get(category) if category else None
        if entity:
            answers = row.get("answers", {}) or {}
            texts = answers.get("text", []) if isinstance(answers, dict) else []
            # оставляем только ответы, реально попавшие в обрезанный текст —
            # иначе нельзя честно требовать от модели их найти
            visible = [t for t in texts if t and t.strip() and t.strip() in by_contract[title].text]
            _add_gold(by_contract[title].gold, entity, visible)

        if len(order) >= n_docs and title == order[-1]:
            # уже набрали n_docs контрактов; добиваем gold только для уже взятых
            pass

    samples = [by_contract[t] for t in order[:n_docs]]
    return samples


def synthetic_subset(n_docs: int = 6) -> list[ContractSample]:
    """Маленькая оффлайн-выборка с известным gold — чтобы код можно было прогнать без сети
    (CI / быстрый тест метрик и пайплайна)."""
    base = [
        (
            "This Non-Disclosure Agreement is made effective as of January 15, 2021, "
            "between Globex Corporation and Jane Doe. The Receiving Party shall not "
            "disclose Confidential Information. This Agreement shall be governed by the "
            "laws of the State of New York. A penalty of $25,000 applies to any breach.",
            {
                "CONTRACT_TYPE": ["Non-Disclosure Agreement"],
                "ORG": ["Globex Corporation"],
                "PERSON": ["Jane Doe"],
                "DATE": ["January 15, 2021"],
                "JURISDICTION": ["State of New York"],
                "MONEY": ["$25,000"],
                "OBLIGATION": ["The Receiving Party shall not disclose Confidential Information"],
            },
        ),
        (
            "This Master Services Agreement (the \"Agreement\") is entered into on "
            "March 3, 2020 by and between Acme Industries Inc. and Initech LLC. "
            "The Supplier shall deliver the services within 30 days. The total fee is "
            "USD 120,000. This Agreement is governed by the laws of California.",
            {
                "CONTRACT_TYPE": ["Master Services Agreement"],
                "ORG": ["Acme Industries Inc.", "Initech LLC"],
                "DATE": ["March 3, 2020"],
                "MONEY": ["USD 120,000"],
                "JURISDICTION": ["California"],
                "OBLIGATION": ["The Supplier shall deliver the services within 30 days"],
            },
        ),
    ]
    samples: list[ContractSample] = []
    for i in range(n_docs):
        text, gold_part = base[i % len(base)]
        gold = {t: [] for t in ENTITY_TYPES}
        gold.update(gold_part)
        samples.append(
            ContractSample(doc_id=f"synthetic-{i}", title=f"synthetic-{i}", text=text, gold=gold)
        )
    return samples


def get_subset(n_docs: int = 100, max_chars: int = 3000, **kw) -> list[ContractSample]:
    """Удобная точка входа: пытается CUAD, при ошибке — синтетика."""
    try:
        samples = load_cuad_subset(n_docs=n_docs, max_chars=max_chars, **kw)
        print(f"[data_prep] CUAD загружен: {len(samples)} контрактов.")
        return samples
    except Exception as e:  # noqa: BLE001
        print(f"[data_prep] CUAD недоступен ({e}); использую synthetic_subset().")
        return synthetic_subset(n_docs=min(n_docs, 6))


def save_subset(samples: list[ContractSample], path: str | Path) -> None:
    path = Path(path)
    path.write_text(
        json.dumps([s.to_dict() for s in samples], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[data_prep] сохранено {len(samples)} документов → {path}")


def load_subset(path: str | Path) -> list[ContractSample]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for d in data:
        gold = {t: [] for t in ENTITY_TYPES}
        gold.update(d.get("gold", {}))
        out.append(ContractSample(d["doc_id"], d["title"], d["text"], gold))
    return out


def gold_coverage(samples: list[ContractSample]) -> dict[str, int]:
    """Сколько gold-спанов всего по каждому типу сущности (диагностика выборки)."""
    cov = {t: 0 for t in ENTITY_TYPES}
    for s in samples:
        for t in ENTITY_TYPES:
            cov[t] += len(s.gold.get(t, []))
    return cov


if __name__ == "__main__":
    print("data_prep — демонстрация (n_docs=20)\n" + "=" * 50)
    samples = get_subset(n_docs=20, max_chars=2500)

    ex = samples[0]
    print("\nПример документа:")
    print("  title:", ex.title)
    print("  text[:300]:", ex.text[:300].replace("\n", " "), "...")
    print("  gold:", {k: v for k, v in ex.gold.items() if v})

    print("\nПокрытие gold по типам (сумма спанов):")
    for t, c in gold_coverage(samples).items():
        print(f"  {t:14s}: {c}")

    out = Path("cuad_subset.json")
    save_subset(samples, out)
    print("\nГотово. Покрытие по категориям с gold:", GOLD_ENTITY_TYPES)
