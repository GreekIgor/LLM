# -*- coding: utf-8 -*-
"""
Бенчмарк производительности извлечения сущностей.

Меряем то, что просит задание (этап 4 «Анализ производительности»):
    • скорость         — tokens/sec и docs/sec;
    • качество         — precision/recall/F1 (через evaluate.py, если есть gold);
    • ресурсы          — пиковая RAM процесса (psutil) и, если есть CUDA, пик VRAM.

Поддерживает:
    • сравнение нескольких моделей (например, Qwen2.5-0.5B vs 1.5B);
    • сравнение batch_size (1 vs 4 vs 8) — показывает выигрыш батчинга на CPU;
    • запуск без gold (тогда считаем только скорость/ресурсы).

Запуск:
    python benchmark.py                       # дефолт: 1 модель, маленькая выборка, CPU
    python benchmark.py --models Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct
    python benchmark.py --n-docs 50 --batch-sizes 1 4 8
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from data_prep import get_subset
from evaluate import evaluate, format_report
from ie_extractor import Extractor


# ─────────────────────────────────────────────────────────────────────────────
# Замер ресурсов
# ─────────────────────────────────────────────────────────────────────────────
def _rss_mb() -> float:
    """Текущее потребление RAM процессом, МБ."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024**2)
    except Exception:  # noqa: BLE001
        return float("nan")


def _cuda_peak_mb() -> float:
    """Пик VRAM, МБ (0, если CUDA нет)."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**2)
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def _cuda_reset_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Результат одного прогона
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RunResult:
    model: str
    dtype: str
    device: str
    batch_size: int
    n_docs: int
    seconds: float
    tokens_per_sec: float
    docs_per_sec: float
    gen_tokens: int
    ram_mb_after_load: float
    ram_mb_peak: float
    vram_mb_peak: float
    metrics: dict[str, Any] = field(default_factory=dict)

    def short(self) -> str:
        f1 = self.metrics.get("micro", {}).get("f1", "—")
        return (
            f"{self.model.split('/')[-1]:<26} dtype={self.dtype:<8} bs={self.batch_size} "
            f"| {self.tokens_per_sec:6.1f} tok/s | {self.docs_per_sec:5.2f} doc/s "
            f"| RAM~{self.ram_mb_peak:6.0f}MB | F1={f1}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Прогон одной модели
# ─────────────────────────────────────────────────────────────────────────────
def benchmark_model(
    model_name: str,
    samples,
    *,
    device: str = "cpu",
    dtype: str = "float32",
    batch_sizes: list[int] = (1,),
    max_new_tokens: int = 384,
    max_input_tokens: int = 1536,
) -> list[RunResult]:
    """Грузит модель один раз и прогоняет её на нескольких batch_size."""
    texts = [s.text for s in samples]
    gold = [s.gold for s in samples]

    print(f"\n>>> Загрузка модели {model_name} (dtype={dtype}, device={device}) ...")
    _cuda_reset_peak()
    ext = Extractor(
        model_name=model_name,
        device=device,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
        max_input_tokens=max_input_tokens,
    )
    t_load = time.perf_counter()
    ext.load()
    print(f"    загружено за {time.perf_counter() - t_load:.1f}s; RAM={_rss_mb():.0f}MB")
    ram_after_load = _rss_mb()

    results: list[RunResult] = []
    for bs in batch_sizes:
        print(f"    прогон batch_size={bs} на {len(texts)} документах ...")
        _cuda_reset_peak()
        preds, stats = ext.extract_batch(texts, batch_size=bs)

        metrics = {}
        try:
            metrics = evaluate(preds, gold)
        except Exception as e:  # noqa: BLE001
            print(f"    (метрики не посчитаны: {e})")

        results.append(
            RunResult(
                model=model_name,
                dtype=dtype,
                device=device,
                batch_size=bs,
                n_docs=len(texts),
                seconds=round(stats["seconds"], 2),
                tokens_per_sec=round(stats["tokens_per_sec"], 2),
                docs_per_sec=round(stats["docs_per_sec"], 3),
                gen_tokens=int(stats["gen_tokens"]),
                ram_mb_after_load=round(ram_after_load, 1),
                ram_mb_peak=round(_rss_mb(), 1),
                vram_mb_peak=round(_cuda_peak_mb(), 1),
                metrics=metrics,
            )
        )
        print("    " + results[-1].short())

    # освобождаем память перед следующей моделью
    del ext
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Бенчмарк извлечения сущностей из CUAD.")
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-1.5B-Instruct"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--n-docs", type=int, default=10, help="размер подвыборки CUAD")
    ap.add_argument("--max-chars", type=int, default=2500, help="обрезка текста контракта")
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1])
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--out", default="benchmark_results.json")
    args = ap.parse_args()

    print("=" * 70)
    print("БЕНЧМАРК ИЗВЛЕЧЕНИЯ СУЩНОСТЕЙ (CUAD)")
    print("=" * 70)
    samples = get_subset(n_docs=args.n_docs, max_chars=args.max_chars)
    print(f"Документов в выборке: {len(samples)}")

    all_results: list[RunResult] = []
    for model_name in args.models:
        all_results += benchmark_model(
            model_name,
            samples,
            device=args.device,
            dtype=args.dtype,
            batch_sizes=args.batch_sizes,
            max_new_tokens=args.max_new_tokens,
        )

    print("\n" + "=" * 70)
    print("СВОДКА")
    print("=" * 70)
    for r in all_results:
        print(r.short())

    # подробные метрики последнего прогона
    if all_results and all_results[-1].metrics:
        print("\nМетрики качества (последний прогон):")
        print(format_report(all_results[-1].metrics))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in all_results], f, ensure_ascii=False, indent=2)
    print(f"\nРезультаты сохранены → {args.out}")


if __name__ == "__main__":
    main()
