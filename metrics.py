"""
Метрики качества для оценки QA-генерации (Трек A).

Реализует:
- normalize_text   — нормализация ответа (lower, пунктуация, артикли/частицы, пробелы)
- exact_match      — точное совпадение после нормализации
- token_f1         — token-level F1 (как в SQuAD)
- compute_bleu     — BLEU через sacrebleu (sentence-level)
- semantic_similarity — косинусная близость эмбеддингов (sentence-transformers)
- bootstrap_ci     — bootstrap-доверительный интервал среднего
- paired_bootstrap_pvalue — парный bootstrap-тест значимости разницы двух моделей

Метрики семантической близости требуют sentence-transformers; остальные — чистый Python
(+ sacrebleu для BLEU), чтобы их можно было прогонять без GPU.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Sequence

import numpy as np

# Русские "стоп-частицы", которые не несут фактической информации в коротком ответе.
# Намеренно короткий список — агрессивная чистка исказила бы EM/F1.
_RU_FILLERS = {"и", "а", "но", "в", "во", "на", "с", "со", "это", "то"}


def normalize_text(s: str) -> str:
    """Привести ответ к каноническому виду для честного сравнения.

    Шаги (по аналогии с официальной нормализацией SQuAD, адаптировано под русский):
    нижний регистр → удаление пунктуации → выкидывание частиц-филлеров →
    схлопывание пробелов.
    """
    if s is None:
        return ""
    s = s.lower()
    # Убираем пунктуацию (латиница + кириллические кавычки/тире).
    s = re.sub(rf"[{re.escape(string.punctuation)}«»—–…]", " ", s)
    tokens = [t for t in s.split() if t not in _RU_FILLERS]
    return " ".join(tokens).strip()


def exact_match(prediction: str, reference: str) -> int:
    """1, если нормализованные строки совпадают полностью, иначе 0."""
    return int(normalize_text(prediction) == normalize_text(reference))


def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 между предсказанием и эталоном (метрика SQuAD).

    Считается как 2*P*R/(P+R) по пересечению мультимножеств токенов.
    """
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_bleu(prediction: str, reference: str) -> float:
    """Sentence-level BLEU (0..100) через sacrebleu.

    sacrebleu сам токенизирует и сглаживает; для коротких ответов BLEU обычно
    занижен, поэтому интерпретируем его только в сравнении между моделями.
    """
    from sacrebleu.metrics import BLEU

    bleu = BLEU(effective_order=True)
    return bleu.sentence_score(prediction or "", [reference or ""]).score


def semantic_similarity(
    predictions: Sequence[str],
    references: Sequence[str],
    model,
) -> np.ndarray:
    """Косинусная близость эмбеддингов пар (pred, ref).

    model — загруженный SentenceTransformer. Возвращает массив значений 0..1.
    Батч-кодирование, чтобы не гонять модель по одному примеру.
    """
    from sentence_transformers import util

    emb_pred = model.encode(list(predictions), convert_to_tensor=True, normalize_embeddings=True)
    emb_ref = model.encode(list(references), convert_to_tensor=True, normalize_embeddings=True)
    cos = util.cos_sim(emb_pred, emb_ref).diagonal()
    return cos.cpu().numpy()


# --------------------------------------------------------------------------- #
# Статистическая значимость (бонус)
# --------------------------------------------------------------------------- #
def bootstrap_ci(values: Sequence[float], n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 42) -> tuple[float, float, float]:
    """Bootstrap-доверительный интервал среднего.

    Возвращает (mean, lo, hi) для уровня (1-alpha).
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    n = len(values)
    boot_means = values[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(lo), float(hi)


def paired_bootstrap_pvalue(scores_a: Sequence[float], scores_b: Sequence[float],
                            n_boot: int = 10_000, seed: int = 42) -> float:
    """Двусторонний p-value парного bootstrap-теста: H0 — средние равны.

    scores_a, scores_b — поэлементные метрики двух моделей на ОДНИХ примерах.
    """
    rng = np.random.default_rng(seed)
    diff = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    n = len(diff)
    observed = diff.mean()
    # Центрируем под H0 и ресэмплируем.
    centered = diff - observed
    boot = centered[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    p = np.mean(np.abs(boot) >= np.abs(observed))
    return float(p)


if __name__ == "__main__":
    # Мини-самопроверка метрик без сети/GPU.
    assert exact_match("Москва.", "москва") == 1
    assert exact_match("Москва", "Питер") == 0
    assert abs(token_f1("столица Россия Москва", "Москва столица") - 0.8) < 1e-9
    assert token_f1("", "") == 1.0
    assert compute_bleu("кошка сидит на столе", "кошка сидит на столе") > 99
    m, lo, hi = bootstrap_ci([1, 0, 1, 1, 0, 1])
    assert lo <= m <= hi
    print("metrics.py: все самопроверки пройдены OK")
