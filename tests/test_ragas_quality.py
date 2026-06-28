"""
Quality-gate тесты на метриках Ragas (ДЗ-19).

Стратегия (важно для CI без секретов):
  * По умолчанию тесты читают УЖЕ СОХРАНЁННЫЙ ragas_results.json (артефакт прогона
    ragas_eval.py) и проверяют, что средние метрики не ниже порогов из thresholds.json.
    Так quality gate работает в CI детерминированно и не требует API-ключа.
  * Если выставлена переменная RAGAS_LIVE=1 (и есть OPENROUTER_API_KEY), тесты сначала
    запускают живой прогон Ragas и проверяют свежие метрики.

Пороги задаются в thresholds.json (например, Faithfulness >= 0.70).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS_JSON = ROOT / "ragas_results.json"
THRESHOLDS = json.loads(
    (ROOT / "thresholds.json").read_text(encoding="utf-8")
)
THRESHOLDS = {k: v for k, v in THRESHOLDS.items() if not k.startswith("_")}
LIVE = os.getenv("RAGAS_LIVE") == "1"


def _maybe_run_live():
    """Если RAGAS_LIVE=1 — перегенерировать ragas_results.json живым прогоном."""
    if not LIVE:
        return
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("RAGAS_LIVE=1, но не задан OPENROUTER_API_KEY")
    # Запускаем оценку как подпроцесс; код возврата нам не важен — гейты проверим сами.
    subprocess.run([sys.executable, str(ROOT / "ragas_eval.py")], cwd=ROOT)


@pytest.fixture(scope="session")
def results():
    _maybe_run_live()
    if not RESULTS_JSON.exists():
        pytest.skip(
            "Нет ragas_results.json. Сначала запусти `python ragas_eval.py` "
            "(нужен OPENROUTER_API_KEY) или используй закоммиченный артефакт."
        )
    return json.loads(RESULTS_JSON.read_text(encoding="utf-8"))


def test_results_have_samples(results):
    assert results.get("n_samples", 0) >= 10, "должно быть не меньше 10 golden-примеров"
    assert results["samples"], "в отчёте нет ни одного примера"


@pytest.mark.parametrize("metric", list(THRESHOLDS.keys()))
def test_metric_above_threshold(results, metric):
    """Среднее значение метрики не должно быть ниже порога (quality gate)."""
    summary = results["summary"]
    assert metric in summary, f"метрика {metric} отсутствует в отчёте"
    value = summary[metric]
    threshold = THRESHOLDS[metric]
    assert value >= threshold, (
        f"{metric} = {value:.3f} ниже порога {threshold:.2f} — возможны галлюцинации "
        f"или плохой контекст"
    )


def test_faithfulness_per_sample_not_collapsing(results):
    """Защита от галлюцинаций: не более 20% примеров с faithfulness ниже порога."""
    thr = THRESHOLDS.get("faithfulness", 0.7)
    vals = [
        s["scores"].get("faithfulness")
        for s in results["samples"]
        if s["scores"].get("faithfulness") is not None
    ]
    assert vals, "нет посчитанных значений faithfulness"
    bad = [v for v in vals if v < thr]
    ratio = len(bad) / len(vals)
    assert ratio <= 0.20, (
        f"{len(bad)}/{len(vals)} примеров с faithfulness < {thr} "
        f"({ratio:.0%}) — слишком много галлюцинаций"
    )


def test_gates_flag_consistent(results):
    """Флаг gates_passed в артефакте должен совпадать с фактической проверкой порогов."""
    summary = results["summary"]
    expected = all(
        summary.get(m, 0) >= t for m, t in THRESHOLDS.items() if m in summary
    )
    assert results.get("gates_passed") == expected
