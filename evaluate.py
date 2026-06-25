# -*- coding: utf-8 -*-
"""
Оценка качества извлечения сущностей: precision / recall / F1.

Особенность задачи:
    Сравнивать предсказанные спаны с gold «строка-в-строку» нечестно — модель пишет
    "Electric City Corp." а gold "Electric City Corp" (без точки), или меняет регистр.
    Поэтому используем НЕЧЁТКОЕ сопоставление (нормализация + перекрытие токенов /
    вхождение подстроки). Это стандартная практика для span-based IE.

Метрики считаем:
    • по каждому типу сущности отдельно (per-type P/R/F1);
    • микро-усреднение по всем типам (общее качество).

Только типы, для которых есть gold (см. data_prep.GOLD_ENTITY_TYPES), участвуют в подсчёте —
по PERSON/MONEY/OBLIGATION в CUAD gold нет, и штрафовать за них некорректно.

Запуск напрямую:  python evaluate.py   — самопроверка метрик на игрушечном примере.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from ie_extractor import ENTITY_TYPES

# ─────────────────────────────────────────────────────────────────────────────
# Нормализация и сопоставление спанов
# ─────────────────────────────────────────────────────────────────────────────
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")
# «шумовые» слова юр-текста, которые не должны мешать матчингу
_STOP = {"the", "a", "an", "of", "inc", "llc", "ltd", "corp", "co", "company", "agreement"}


def normalize(s: str) -> str:
    """lowercase, без диакритики, без пунктуации, схлопнутые пробелы."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _tokens(s: str) -> set[str]:
    return {t for t in normalize(s).split() if t and t not in _STOP}


def spans_match(pred: str, gold: str, threshold: float = 0.5) -> bool:
    """True, если спаны «достаточно похожи».

    Считаем совпадением, если после нормализации:
      • одна строка является подстрокой другой, ИЛИ
      • Jaccard по значимым токенам >= threshold."""
    np_, ng = normalize(pred), normalize(gold)
    if not np_ or not ng:
        return False
    if np_ == ng or np_ in ng or ng in np_:
        return True
    tp, tg = _tokens(pred), _tokens(gold)
    if not tp or not tg:
        return False
    inter = len(tp & tg)
    union = len(tp | tg)
    return (inter / union) >= threshold if union else False


# ─────────────────────────────────────────────────────────────────────────────
# Подсчёт TP/FP/FN с жадным сопоставлением (один gold ↦ один pred)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: "Counts") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def _count_one(preds: list[str], golds: list[str], threshold: float) -> Counts:
    """TP/FP/FN для одного документа и одного типа сущности (жадный матчинг)."""
    matched_gold = [False] * len(golds)
    tp = 0
    for p in preds:
        for gi, g in enumerate(golds):
            if not matched_gold[gi] and spans_match(p, g, threshold):
                matched_gold[gi] = True
                tp += 1
                break
    fp = len(preds) - tp
    fn = matched_gold.count(False)
    return Counts(tp=tp, fp=fp, fn=fn)


# ─────────────────────────────────────────────────────────────────────────────
# Полная оценка по списку документов
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(
    predictions: list[dict[str, list[str]]],
    gold: list[dict[str, list[str]]],
    entity_types: list[str] | None = None,
    threshold: float = 0.5,
) -> dict:
    """Считает per-type и micro P/R/F1.

    entity_types — какие типы учитывать. По умолчанию берём только те, где в gold есть
    хотя бы один спан (иначе тип не из чего оценивать)."""
    assert len(predictions) == len(gold), "preds и gold должны быть одной длины"

    if entity_types is None:
        present = set()
        for g in gold:
            for t in ENTITY_TYPES:
                if g.get(t):
                    present.add(t)
        entity_types = [t for t in ENTITY_TYPES if t in present]

    per_type: dict[str, Counts] = {t: Counts() for t in entity_types}
    for pred_doc, gold_doc in zip(predictions, gold):
        for t in entity_types:
            c = _count_one(pred_doc.get(t, []), gold_doc.get(t, []), threshold)
            per_type[t].add(c)

    micro = Counts()
    for c in per_type.values():
        micro.add(c)

    return {
        "entity_types": entity_types,
        "per_type": {
            t: {
                "precision": round(c.precision, 4),
                "recall": round(c.recall, 4),
                "f1": round(c.f1, 4),
                "tp": c.tp,
                "fp": c.fp,
                "fn": c.fn,
            }
            for t, c in per_type.items()
        },
        "micro": {
            "precision": round(micro.precision, 4),
            "recall": round(micro.recall, 4),
            "f1": round(micro.f1, 4),
            "tp": micro.tp,
            "fp": micro.fp,
            "fn": micro.fn,
        },
    }


def format_report(metrics: dict) -> str:
    """Человекочитаемая таблица метрик."""
    lines = []
    header = f"{'ENTITY':<14} {'P':>7} {'R':>7} {'F1':>7} {'TP':>5} {'FP':>5} {'FN':>5}"
    lines.append(header)
    lines.append("-" * len(header))
    for t, m in metrics["per_type"].items():
        lines.append(
            f"{t:<14} {m['precision']:>7.3f} {m['recall']:>7.3f} {m['f1']:>7.3f} "
            f"{m['tp']:>5} {m['fp']:>5} {m['fn']:>5}"
        )
    lines.append("-" * len(header))
    mi = metrics["micro"]
    lines.append(
        f"{'MICRO':<14} {mi['precision']:>7.3f} {mi['recall']:>7.3f} {mi['f1']:>7.3f} "
        f"{mi['tp']:>5} {mi['fp']:>5} {mi['fn']:>5}"
    )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Самопроверка
# ─────────────────────────────────────────────────────────────────────────────
def _selftest() -> None:
    print("Self-test evaluate()\n" + "=" * 50)

    # матчинг
    assert spans_match("Electric City Corp.", "Electric City Corp")
    assert spans_match("the State of New York", "State of New York")
    assert spans_match("Acme Industries Inc.", "Acme Industries, Inc")
    assert not spans_match("Acme Corp", "Globex LLC")
    print("[OK] spans_match: эквивалентные/разные строки различаются корректно")

    preds = [
        {"ORG": ["Acme Corp.", "Globex"], "DATE": ["March 3, 2020"], "CONTRACT_TYPE": ["NDA"]},
        {"ORG": ["Initech"], "DATE": [], "CONTRACT_TYPE": ["Lease"]},
    ]
    gold = [
        {"ORG": ["Acme Corp"], "DATE": ["March 3 2020"], "CONTRACT_TYPE": ["Services Agreement"]},
        {"ORG": ["Initech LLC"], "DATE": ["Jan 1"], "CONTRACT_TYPE": ["Lease Agreement"]},
    ]
    # дополним до полной схемы
    for d in preds + gold:
        for t in ENTITY_TYPES:
            d.setdefault(t, [])

    m = evaluate(preds, gold)
    print("\nУчтённые типы:", m["entity_types"])
    print(format_report(m))

    # проверим арифметику по ORG: Acme✓, Globex✗(fp), Initech✓ → tp=2 fp=1 fn=0
    org = m["per_type"]["ORG"]
    assert (org["tp"], org["fp"], org["fn"]) == (2, 1, 0), org
    print("\n[OK] арифметика TP/FP/FN по ORG верна")
    print("=" * 50, "\nГОТОВО [OK]")


if __name__ == "__main__":
    _selftest()
