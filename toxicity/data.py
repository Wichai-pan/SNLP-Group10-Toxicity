from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class Example:
    id: str
    text: str
    label: Optional[int]
    lang: str


def infer_lang(example_id: str) -> str:
    return example_id.split("_", 1)[0] if example_id else "unk"


def read_tsv(path: str | Path, *, has_label: bool) -> list[Example]:
    path = Path(path)
    examples: list[Example] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"id", "text"} | ({"label"} if has_label else set())
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        for row in reader:
            example_id = (row.get("id") or "").strip()
            text = row.get("text") or ""
            lang = infer_lang(example_id)
            label = None
            if has_label:
                raw = row.get("label")
                if raw is None:
                    raise ValueError(f"{path}: missing label for id={example_id}")
                label = int(raw)
            examples.append(Example(id=example_id, text=text, label=label, lang=lang))
    return examples


def iter_texts(examples: Iterable[Example]) -> list[str]:
    return [ex.text for ex in examples]


def iter_labels(examples: Iterable[Example]) -> list[int]:
    labels: list[int] = []
    for ex in examples:
        if ex.label is None:
            raise ValueError("Encountered example without label.")
        labels.append(int(ex.label))
    return labels


def iter_ids(examples: Iterable[Example]) -> list[str]:
    return [ex.id for ex in examples]


def iter_langs(examples: Iterable[Example]) -> list[str]:
    return [ex.lang for ex in examples]

