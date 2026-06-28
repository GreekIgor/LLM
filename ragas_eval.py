"""
Оценка RAG-бота с помощью Ragas (ДЗ-19).

Метрики:
  * Faithfulness     — насколько ответ опирается на контекст (проверка на ГАЛЛЮЦИНАЦИИ);
  * Answer Relevance — насколько ответ релевантен заданному вопросу;
  * Context Recall   — насколько извлечённый контекст покрывает эталонный ответ (reference).

Что делает скрипт:
  1. читает золотые примеры (tests/goldens.json);
  2. для каждого вопроса прогоняет RAG-приложение (retrieve + generate через OpenRouter);
  3. считает метрики Ragas (LLM-судья — тоже OpenRouter, эмбеддинги — fastembed);
  4. сохраняет результаты в ragas_results.json и человекочитаемый ragas_report.html;
  5. печатает сравнение средних метрик с порогами (thresholds.json) и кодом возврата
     сигналит CI: 0 — все гейты пройдены, 1 — есть просадка.

Запуск:  python ragas_eval.py            # полный прогон (нужен OPENROUTER_API_KEY)
         python ragas_eval.py --limit 3  # быстрый прогон по 3 примерам

ВНИМАНИЕ: шаг требует обращения к LLM-API и интернета. В CI без секрета этот скрипт
пропускается, а quality gate проверяется по уже сохранённому ragas_results.json (см. README).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

ROOT = Path(__file__).parent
GOLDENS_PATH = ROOT / "tests" / "goldens.json"
THRESHOLDS_PATH = ROOT / "thresholds.json"
RESULTS_JSON = ROOT / "ragas_results.json"
REPORT_HTML = ROOT / "ragas_report.html"

# Имена метрик, которые мы используем (ключи в результатах).
METRIC_KEYS = ["faithfulness", "answer_relevancy", "context_recall"]


def load_goldens(limit: int | None = None) -> list[dict]:
    data = json.loads(GOLDENS_PATH.read_text(encoding="utf-8"))
    items = data["items"]
    return items[:limit] if limit else items


def load_thresholds() -> dict:
    data = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def build_evaluation_dataset(goldens: list[dict]):
    """Прогоняет RAG по каждому вопросу и собирает EvaluationDataset для Ragas."""
    from ragas import EvaluationDataset

    from rag_app import Retriever, generate_answer

    retriever = Retriever()  # единый индекс на все вопросы
    rows = []
    for i, g in enumerate(goldens, 1):
        print(f"  [{i}/{len(goldens)}] {g['question']}")
        out = generate_answer(g["question"], retriever=retriever)
        rows.append(
            {
                "user_input": g["question"],
                "response": out["answer"],
                "retrieved_contexts": out["contexts"],
                "reference": g["ground_truth"],
            }
        )
    return EvaluationDataset.from_list(rows), rows


def run_ragas(dataset):
    """Считает три метрики Ragas. Возвращает (result, pandas.DataFrame)."""
    from ragas import evaluate
    from ragas.metrics import Faithfulness, LLMContextRecall, ResponseRelevancy

    from config import get_ragas_embeddings, get_ragas_llm

    evaluator_llm = get_ragas_llm()
    evaluator_emb = get_ragas_embeddings()

    metrics = [
        Faithfulness(),          # -> ключ "faithfulness"
        ResponseRelevancy(),     # -> ключ "answer_relevancy"
        LLMContextRecall(),      # -> ключ "context_recall"
    ]
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_emb,
    )
    return result, result.to_pandas()


def summarize(df) -> dict:
    """Средние значения по каждой метрике (NaN игнорируются)."""
    summary = {}
    for key in METRIC_KEYS:
        if key in df.columns:
            summary[key] = float(df[key].mean(skipna=True))
    return summary


def check_gates(summary: dict, thresholds: dict) -> tuple[bool, list[str]]:
    """Сверяет средние метрики с порогами. Возвращает (passed, messages)."""
    passed = True
    msgs = []
    for key, thr in thresholds.items():
        value = summary.get(key)
        if value is None:
            msgs.append(f"[SKIP] {key}: метрика не посчитана")
            continue
        ok = value >= thr
        passed = passed and ok
        flag = "OK  " if ok else "FAIL"
        msgs.append(f"[{flag}] {key}: {value:.3f} (порог >= {thr:.2f})")
    return passed, msgs


def save_json(summary, thresholds, per_sample_rows, df, gates_passed):
    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "metrics_used": METRIC_KEYS,
        "thresholds": thresholds,
        "summary": summary,
        "gates_passed": gates_passed,
        "n_samples": len(per_sample_rows),
        "samples": [],
    }
    records = df.to_dict(orient="records")
    for row, rec in zip(per_sample_rows, records):
        payload["samples"].append(
            {
                "question": row["user_input"],
                "answer": row["response"],
                "reference": row["reference"],
                "contexts": row["retrieved_contexts"],
                "scores": {
                    k: (float(rec[k]) if k in rec and rec[k] == rec[k] else None)
                    for k in METRIC_KEYS
                },
            }
        )
    RESULTS_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[*] JSON-отчёт сохранён: {RESULTS_JSON.name}")
    return payload


def save_html(payload):
    """Простой самодостаточный HTML-отчёт без внешних зависимостей."""
    def cell(v):
        if v is None:
            return '<td class="na">n/a</td>'
        cls = "good" if v >= 0.7 else "bad"
        return f'<td class="{cls}">{v:.3f}</td>'

    rows_html = ""
    for s in payload["samples"]:
        sc = s["scores"]
        rows_html += (
            "<tr>"
            f"<td class='q'>{_esc(s['question'])}</td>"
            f"<td>{_esc(s['answer'])}</td>"
            f"{cell(sc.get('faithfulness'))}"
            f"{cell(sc.get('answer_relevancy'))}"
            f"{cell(sc.get('context_recall'))}"
            "</tr>"
        )

    summary = payload["summary"]
    thr = payload["thresholds"]
    summary_rows = ""
    for k in METRIC_KEYS:
        val = summary.get(k)
        t = thr.get(k)
        ok = val is not None and t is not None and val >= t
        badge = "PASS" if ok else "FAIL"
        bcls = "good" if ok else "bad"
        vtxt = f"{val:.3f}" if val is not None else "n/a"
        summary_rows += (
            f"<tr><td>{k}</td><td>{vtxt}</td><td>{t}</td>"
            f"<td class='{bcls}'>{badge}</td></tr>"
        )

    overall = "ПРОЙДЕН" if payload["gates_passed"] else "ПРОВАЛЕН"
    overall_cls = "good" if payload["gates_passed"] else "bad"

    html = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<title>Ragas отчёт — ДЗ-19</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Arial, sans-serif; margin: 24px; color:#1a1a1a; }}
  h1 {{ font-size: 22px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; font-size: 14px; }}
  th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
  th {{ background: #f4f4f4; }}
  td.good {{ background: #e6f6e6; color:#0a6b0a; font-weight:600; }}
  td.bad {{ background: #fdeaea; color:#b00020; font-weight:600; }}
  td.na {{ color:#888; }}
  td.q {{ max-width: 320px; }}
  .gate {{ font-size: 18px; font-weight: 700; }}
  .good {{ color:#0a6b0a; }} .bad {{ color:#b00020; }}
  .meta {{ color:#666; font-size: 13px; }}
</style></head><body>
<h1>Отчёт Ragas: оценка RAG-бота (ДЗ-19)</h1>
<p class="meta">Сгенерировано: {payload['generated_at']} &middot; примеров: {payload['n_samples']}</p>
<p class="gate">Quality gate: <span class="{overall_cls}">{overall}</span></p>

<h2>Сводка по метрикам</h2>
<table>
  <tr><th>Метрика</th><th>Среднее</th><th>Порог</th><th>Статус</th></tr>
  {summary_rows}
</table>

<h2>Подробно по примерам</h2>
<table>
  <tr><th>Вопрос</th><th>Ответ модели</th><th>Faithfulness</th><th>Answer&nbsp;Relevance</th><th>Context&nbsp;Recall</th></tr>
  {rows_html}
</table>
</body></html>"""
    REPORT_HTML.write_text(html, encoding="utf-8")
    print(f"[*] HTML-отчёт сохранён: {REPORT_HTML.name}")


def _esc(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def main():
    parser = argparse.ArgumentParser(description="Ragas-оценка RAG-бота (ДЗ-19)")
    parser.add_argument("--limit", type=int, default=None, help="ограничить число примеров")
    args = parser.parse_args()

    thresholds = load_thresholds()
    goldens = load_goldens(limit=args.limit)
    print(f"[*] Золотых примеров к прогону: {len(goldens)}\n")

    print("[*] Генерация ответов RAG (retrieve + LLM):")
    dataset, rows = build_evaluation_dataset(goldens)

    print("\n[*] Подсчёт метрик Ragas (LLM-судья + эмбеддинги)...")
    result, df = run_ragas(dataset)

    summary = summarize(df)
    gates_passed, msgs = check_gates(summary, thresholds)

    print("\n=== Quality gates ===")
    for m in msgs:
        print(" " + m)

    payload = save_json(summary, thresholds, rows, df, gates_passed)
    save_html(payload)

    print(f"\n[{'OK' if gates_passed else 'FAIL'}] Итог: гейты "
          f"{'пройдены' if gates_passed else 'НЕ пройдены'}.")
    raise SystemExit(0 if gates_passed else 1)


if __name__ == "__main__":
    main()
