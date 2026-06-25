# -*- coding: utf-8 -*-
"""
Demo-приложение: интерактивное извлечение сущностей из текста контракта.

Что делает:
    Вставляешь текст договора → модель возвращает сущности по 7 типам (PERSON, ORG, MONEY,
    DATE, CONTRACT_TYPE, OBLIGATION, JURISDICTION). Показываем красивую разметку
    (HighlightedText) + структурированный JSON, который можно скопировать в свою систему.

Запуск:
    python app.py                       # модель по умолчанию Qwen2.5-1.5B-Instruct (CPU)
    IE_MODEL=Qwen/Qwen2.5-0.5B-Instruct python app.py   # быстрее, чуть хуже качество

Модель грузится один раз при старте (на CPU это десятки секунд; первое извлечение тоже
небыстрое). Это нормально для локального CPU — для скорости используйте Colab/GPU.
"""
from __future__ import annotations

import os

from ie_extractor import ENTITY_TYPES, Extractor

MODEL_NAME = os.environ.get("IE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

# Цвет-метки для HighlightedText (Gradio сам подберёт палитру по ключам).
EXAMPLES = [
    "This Non-Disclosure Agreement is made effective as of January 15, 2021, between "
    "Globex Corporation and Jane Doe. The Receiving Party shall not disclose Confidential "
    "Information. This Agreement shall be governed by the laws of the State of New York. "
    "A penalty of $25,000 applies to any breach.",
    "This Master Services Agreement is entered into on March 3, 2020 by and between "
    "Acme Industries Inc. and Initech LLC. The Supplier shall deliver the services within "
    "30 days. The total fee is USD 120,000. This Agreement is governed by the laws of California.",
]

# Ленивая инициализация — модель грузим при первом обращении.
_extractor: Extractor | None = None


def _get_extractor() -> Extractor:
    global _extractor
    if _extractor is None:
        print(f"[app] загружаю модель {MODEL_NAME} (CPU) — подождите ...")
        _extractor = Extractor(model_name=MODEL_NAME, device="cpu").load()
        print("[app] модель готова.")
    return _extractor


def _to_highlighted(text: str, entities: dict[str, list[str]]) -> list[tuple[str, str | None]]:
    """Размечает исходный текст для gr.HighlightedText: список (фрагмент, метка|None).

    Идём по тексту и подсвечиваем первые вхождения извлечённых спанов. Для пересечений
    берём самое раннее совпадение. OBLIGATION часто длинная — она тоже подсветится."""
    # собираем (start, end, label), ищем спаны без учёта регистра
    spans: list[tuple[int, int, str]] = []
    low = text.lower()
    for label, values in entities.items():
        for v in values:
            v = v.strip()
            if not v:
                continue
            idx = low.find(v.lower())
            if idx != -1:
                spans.append((idx, idx + len(v), label))
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    # убираем перекрытия (жадно слева направо)
    chosen: list[tuple[int, int, str]] = []
    last_end = -1
    for s, e, lab in spans:
        if s >= last_end:
            chosen.append((s, e, lab))
            last_end = e

    out: list[tuple[str, str | None]] = []
    cur = 0
    for s, e, lab in chosen:
        if s > cur:
            out.append((text[cur:s], None))
        out.append((text[s:e], lab))
        cur = e
    if cur < len(text):
        out.append((text[cur:], None))
    return out or [(text, None)]


def extract(text: str):
    """Колбэк кнопки: текст → (подсветка, JSON-словарь)."""
    text = (text or "").strip()
    if not text:
        return [("Введите текст контракта…", None)], {}
    ext = _get_extractor()
    entities = ext.extract(text)
    highlighted = _to_highlighted(text, entities)
    # отдаём только непустые типы в JSON для читаемости, но полную схему оставляем доступной
    return highlighted, entities


def build_ui():
    import gradio as gr

    with gr.Blocks(title="Contract Entity Extraction") as demo:
        gr.Markdown(
            f"# 📄 Извлечение сущностей из контрактов\n"
            f"Модель: **{MODEL_NAME}** (локально, CPU). Типы сущностей: "
            f"{', '.join(ENTITY_TYPES)}."
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Textbox(label="Текст контракта", lines=12, placeholder="Вставьте текст…")
                btn = gr.Button("Извлечь сущности", variant="primary")
                gr.Examples(EXAMPLES, inputs=inp)
            with gr.Column():
                hl = gr.HighlightedText(label="Размеченный текст", combine_adjacent=True)
                js = gr.JSON(label="Сущности (JSON)")
        btn.click(extract, inputs=inp, outputs=[hl, js])
    return demo


if __name__ == "__main__":
    build_ui().launch()
