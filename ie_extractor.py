# -*- coding: utf-8 -*-
"""
Ядро извлечения сущностей (Information Extraction) из текстов контрактов.

Идея:
    Вместо обучения отдельной NER-модели мы используем уже готовую инструктивную LLM
    в режиме zero-/few-shot: даём ей текст и просим вернуть СТРОГИЙ JSON со списками
    найденных сущностей по фиксированной схеме. Это «LLM-as-extractor» — гибко
    (новый тип сущности = одна строчка в промпте), но требует аккуратного парсинга
    ответа модели, потому что маленькие модели любят добавить лишний текст вокруг JSON.

Схема сущностей (7 типов из задания):
    PERSON         — физические лица (ФИО, подписанты);
    ORG            — организации, компании, стороны договора;
    MONEY          — денежные суммы (с валютой);
    DATE           — даты (подписания, вступления в силу, истечения);
    CONTRACT_TYPE  — тип/название договора (NDA, Lease, Service Agreement ...);
    OBLIGATION     — обязательства сторон (краткие формулировки «кто что должен»);
    JURISDICTION   — применимое право / юрисдикция (Governing Law).

Файл самодостаточный:
    • импортируй     →  from ie_extractor import Extractor, ENTITY_TYPES, parse_entities
    • запусти напрямую →  python ie_extractor.py   (быстрый offline-тест парсера без модели)
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Схема сущностей
# ─────────────────────────────────────────────────────────────────────────────
ENTITY_TYPES: list[str] = [
    "PERSON",
    "ORG",
    "MONEY",
    "DATE",
    "CONTRACT_TYPE",
    "OBLIGATION",
    "JURISDICTION",
]

_ENTITY_DESCRIPTIONS: dict[str, str] = {
    "PERSON": "individual people / natural persons (full names, signatories)",
    "ORG": "organizations, companies, the contracting parties",
    "MONEY": "monetary amounts including currency (e.g. '$10,000', 'USD 5 million')",
    "DATE": "dates (signing / effective / expiration dates)",
    "CONTRACT_TYPE": "the type or title of the contract (e.g. 'Non-Disclosure Agreement')",
    "OBLIGATION": "obligations: short 'who must do what' statements",
    "JURISDICTION": "governing law / jurisdiction (e.g. 'State of California')",
}

# Пустой результат правильной формы — используем как fallback и как «контракт» формата.
EMPTY_RESULT: dict[str, list[str]] = {t: [] for t in ENTITY_TYPES}


# ─────────────────────────────────────────────────────────────────────────────
# Построение промпта
# ─────────────────────────────────────────────────────────────────────────────
def build_messages(text: str, *, few_shot: bool = True) -> list[dict[str, str]]:
    """Собирает chat-сообщения для инструктивной модели.

    Возвращаем список вида [{role, content}, ...], который дальше прогоняется через
    tokenizer.apply_chat_template(). few_shot=True добавляет один пример — это заметно
    стабилизирует формат у маленьких моделей (1.5B и меньше)."""
    schema_lines = "\n".join(f'  - {t}: {_ENTITY_DESCRIPTIONS[t]}' for t in ENTITY_TYPES)
    keys = ", ".join(f'"{t}"' for t in ENTITY_TYPES)

    system = (
        "You are a precise information-extraction engine for legal contracts. "
        "Extract entities and return ONLY a single JSON object, no prose, no markdown.\n"
        f"The JSON MUST have exactly these keys: {keys}.\n"
        "Each value is a JSON array of strings (verbatim spans from the text). "
        "Use an empty array [] when nothing of that type is present. "
        "Do not invent entities that are not in the text.\n"
        "Entity types:\n" + schema_lines
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]

    if few_shot:
        example_text = (
            "This Services Agreement is entered into as of March 3, 2020 by and between "
            "Acme Corp. and John Smith. The Client shall pay $50,000. "
            "This Agreement shall be governed by the laws of the State of Delaware."
        )
        example_json = {
            "PERSON": ["John Smith"],
            "ORG": ["Acme Corp."],
            "MONEY": ["$50,000"],
            "DATE": ["March 3, 2020"],
            "CONTRACT_TYPE": ["Services Agreement"],
            "OBLIGATION": ["The Client shall pay $50,000"],
            "JURISDICTION": ["State of Delaware"],
        }
        messages.append({"role": "user", "content": f"TEXT:\n{example_text}"})
        messages.append(
            {"role": "assistant", "content": json.dumps(example_json, ensure_ascii=False)}
        )

    messages.append({"role": "user", "content": f"TEXT:\n{text}"})
    return messages


# ─────────────────────────────────────────────────────────────────────────────
# Парсинг ответа модели  →  dict[str, list[str]]
# ─────────────────────────────────────────────────────────────────────────────
def _first_json_object(s: str) -> str | None:
    """Вырезает первый сбалансированный по фигурным скобкам JSON-объект из строки.

    Маленькие модели часто оборачивают JSON в ```json ... ``` или дописывают
    комментарии. Идём по символам и считаем баланс скобок, игнорируя скобки внутри строк."""
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
    return None


def _normalize(raw: Any) -> dict[str, list[str]]:
    """Приводит произвольный распарсенный объект к строгой схеме EMPTY_RESULT."""
    out = {t: [] for t in ENTITY_TYPES}
    if not isinstance(raw, dict):
        return out
    for key, val in raw.items():
        k = str(key).strip().upper().replace(" ", "_")
        if k not in out:
            continue
        if isinstance(val, str):
            items = [val]
        elif isinstance(val, (list, tuple)):
            items = val
        else:
            items = [str(val)]
        # чистим: только непустые строки, без дублей, сохраняя порядок
        seen = set()
        cleaned = []
        for it in items:
            sval = str(it).strip()
            if sval and sval.lower() not in {"none", "null", "n/a"} and sval not in seen:
                seen.add(sval)
                cleaned.append(sval)
        out[k] = cleaned
    return out


def parse_entities(model_output: str) -> dict[str, list[str]]:
    """Главный парсер: из сырого текста модели делает dict схемы ENTITY_TYPES.

    Никогда не бросает исключение — при неудаче возвращает EMPTY_RESULT-подобную форму,
    чтобы batch-обработка не падала на одном плохом ответе."""
    if not model_output:
        return {t: [] for t in ENTITY_TYPES}
    chunk = _first_json_object(model_output)
    if chunk is not None:
        try:
            return _normalize(json.loads(chunk))
        except json.JSONDecodeError:
            pass
    # запасной путь: построчно вытащить "KEY": [ ... ] регуляркой
    out = {t: [] for t in ENTITY_TYPES}
    for t in ENTITY_TYPES:
        m = re.search(rf'"{t}"\s*:\s*\[(.*?)\]', model_output, re.DOTALL | re.IGNORECASE)
        if m:
            vals = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
            out[t] = [v.strip() for v in vals if v.strip()]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Обёртка над локальной LLM (CPU-friendly)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Extractor:
    """Ленивая обёртка над инструктивной HF-моделью для извлечения сущностей.

    Пример:
        ext = Extractor("Qwen/Qwen2.5-1.5B-Instruct")
        ext.load()
        result = ext.extract("This NDA is between Acme Corp and Bob ...")

    Тяжёлые импорты (torch/transformers) делаются только в load(), поэтому модуль можно
    импортировать и использовать parse_entities() без установленного torch."""

    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    device: str = "cpu"
    dtype: str = "float32"          # на CPU bf16/fp16 обычно медленнее или не поддержаны
    max_new_tokens: int = 512
    few_shot: bool = True
    max_input_tokens: int = 2048    # обрезаем длинные контракты, чтобы влезть в контекст/память

    model: Any = field(default=None, repr=False)
    tokenizer: Any = field(default=None, repr=False)

    # ── загрузка ──────────────────────────────────────────────────────────────
    def load(self) -> "Extractor":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        torch_dtype = dtype_map.get(self.dtype, torch.float32)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # для батч-генерации decoder-only моделей нужен left padding
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map=self.device if self.device != "cpu" else None,
        )
        if self.device == "cpu":
            self.model = self.model.to("cpu")
        self.model.eval()
        return self

    def _ensure_loaded(self) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Модель не загружена — вызови .load() перед extract().")

    def _render_prompt(self, text: str) -> str:
        messages = build_messages(text, few_shot=self.few_shot)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # ── одиночное извлечение ───────────────────────────────────────────────────
    def extract(self, text: str) -> dict[str, list[str]]:
        out, _ = self.extract_with_stats(text)
        return out

    def extract_with_stats(self, text: str) -> tuple[dict[str, list[str]], dict[str, float]]:
        """Извлечение + статистика (сгенерировано токенов, секунды, tokens/sec)."""
        import torch

        self._ensure_loaded()
        prompt = self._render_prompt(text)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_input_tokens
        ).to(self.model.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            gen = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        dt = time.perf_counter() - t0

        new_tokens = gen[0][inputs["input_ids"].shape[1] :]
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        stats = {
            "gen_tokens": int(new_tokens.shape[0]),
            "seconds": dt,
            "tokens_per_sec": (int(new_tokens.shape[0]) / dt) if dt > 0 else 0.0,
        }
        return parse_entities(decoded), stats

    # ── батч-извлечение (выше throughput на CPU за счёт параллельной генерации) ─
    def extract_batch(
        self, texts: list[str], batch_size: int = 4
    ) -> tuple[list[dict[str, list[str]]], dict[str, float]]:
        """Обрабатывает список текстов пачками. Возвращает (результаты, агрегат-статистика)."""
        import torch

        self._ensure_loaded()
        results: list[dict[str, list[str]]] = []
        total_new_tokens = 0
        t0 = time.perf_counter()

        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            prompts = [self._render_prompt(t) for t in chunk]
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens,
            ).to(self.model.device)

            with torch.no_grad():
                gen = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            in_len = inputs["input_ids"].shape[1]
            for row in gen:
                new = row[in_len:]
                total_new_tokens += int((new != self.tokenizer.pad_token_id).sum())
                decoded = self.tokenizer.decode(new, skip_special_tokens=True)
                results.append(parse_entities(decoded))

        dt = time.perf_counter() - t0
        stats = {
            "n_docs": len(texts),
            "batch_size": batch_size,
            "seconds": dt,
            "gen_tokens": total_new_tokens,
            "tokens_per_sec": (total_new_tokens / dt) if dt > 0 else 0.0,
            "docs_per_sec": (len(texts) / dt) if dt > 0 else 0.0,
        }
        return results, stats


# ─────────────────────────────────────────────────────────────────────────────
# Offline-самопроверка парсера (без модели и без torch)
# ─────────────────────────────────────────────────────────────────────────────
def _selftest() -> None:
    print("Self-test parse_entities() — без модели\n" + "=" * 50)

    cases = {
        "чистый JSON": '{"PERSON": ["Bob"], "ORG": ["Acme"], "MONEY": ["$5"], '
        '"DATE": [], "CONTRACT_TYPE": ["NDA"], "OBLIGATION": [], "JURISDICTION": ["Texas"]}',
        "JSON в markdown-блоке": '```json\n{"PERSON": ["Ann"], "ORG": ["IBM"]}\n```',
        "JSON + болтовня вокруг": 'Sure! Here is the result:\n'
        '{"ORG": ["Globex"], "DATE": ["2021-01-01"]}\nHope this helps.',
        "строка вместо списка": '{"JURISDICTION": "California", "PERSON": "John Doe"}',
        "битый JSON (fallback regex)": 'junk "PERSON": ["X", "Y"] more junk "ORG": ["Z"] ...',
        "мусор без JSON": "no entities here at all",
    }
    ok = True
    for name, raw in cases.items():
        res = parse_entities(raw)
        valid_shape = set(res.keys()) == set(ENTITY_TYPES) and all(
            isinstance(v, list) for v in res.values()
        )
        nonempty = {k: v for k, v in res.items() if v}
        ok &= valid_shape
        print(f"[{'OK' if valid_shape else 'FAIL'}] {name}: {nonempty}")

    print("=" * 50)
    print("Итог:", "ВСЕ ТЕСТЫ ФОРМЫ ПРОЙДЕНЫ [OK]" if ok else "ЕСТЬ ОШИБКИ [FAIL]")


if __name__ == "__main__":
    _selftest()
