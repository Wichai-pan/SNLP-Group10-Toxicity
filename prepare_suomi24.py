"""
Prepare TurkuNLP Suomi24 Finnish toxicity data for training.

Source : https://huggingface.co/datasets/TurkuNLP/Suomi24-toxicity-annotated
License: CC BY-SA 4.0
Size   : ~2,300 native Finnish comments with multi-label toxicity annotations

Label mapping (binary toxic):
  toxic = 1  if TOXICITY >= 0.5 OR SEVERE_TOXICITY >= 0.5
  toxic = 0  otherwise

Output:
  fin_suomi24.tsv          — Suomi24 rows only (Finnish native data)
  train_aug_suomi24.tsv    — train_aug.tsv (EN+DE+FI translated) + Suomi24
"""
from __future__ import annotations

import csv
from pathlib import Path


def load_suomi24(toxic_threshold: float = 0.5) -> list[dict]:
    """Download Suomi24 via HuggingFace datasets library."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Install the 'datasets' package: pip install datasets\n"
            "Or on Puhti: pip install --user datasets"
        )

    print("Loading TurkuNLP/Suomi24-toxicity-annotated from HuggingFace ...")
    ds = load_dataset("TurkuNLP/Suomi24-toxicity-annotated", trust_remote_code=True)

    rows = []
    split_counts: dict[str, int] = {}
    for split_name, split_data in ds.items():
        count = 0
        for i, ex in enumerate(split_data):
            text = ex.get("text") or ex.get("comment_text") or ""
            if not text:
                continue
            # Binary label: toxic if any primary score >= threshold
            tox_score = float(ex.get("TOXICITY", 0) or 0)
            sev_score = float(ex.get("SEVERE_TOXICITY", 0) or 0)
            label = 1 if (tox_score >= toxic_threshold or sev_score >= toxic_threshold) else 0
            rows.append({"id": f"fin_suomi24_{split_name}_{i}", "text": text, "label": label})
            count += 1
        split_counts[split_name] = count

    toxic = sum(r["label"] for r in rows)
    print(f"  Suomi24: {len(rows)} rows across {list(split_counts)} "
          f"(toxic={toxic}, clean={len(rows)-toxic})")
    return rows


def read_tsv(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append({"id": row["id"], "text": row["text"], "label": int(row["label"])})
    return rows


def write_tsv(rows: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "text", "label"])
        for row in rows:
            w.writerow([row["id"], row["text"], row["label"]])
    print(f"Wrote {len(rows)} rows -> {path}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_tsv",       default="train_aug.tsv",
                    help="Translate-train augmented data (EN+DE+FI translated)")
    ap.add_argument("--out_suomi24",    default="fin_suomi24.tsv",
                    help="Standalone Suomi24 output file")
    ap.add_argument("--out_merged",     default="train_aug_suomi24.tsv",
                    help="train_aug.tsv + Suomi24 merged")
    ap.add_argument("--toxic_threshold", type=float, default=0.5)
    args = ap.parse_args()

    suomi24_rows = load_suomi24(args.toxic_threshold)

    # Save standalone file
    write_tsv(suomi24_rows, args.out_suomi24)

    # Merge with translate-train augmented data
    print(f"\nLoading base: {args.base_tsv}")
    base = read_tsv(args.base_tsv)
    print(f"  Base rows: {len(base)}")

    merged = base + suomi24_rows
    toxic  = sum(1 for r in merged if r["label"] == 1)
    print(f"Merged: {len(merged)} total  (toxic={toxic}, clean={len(merged)-toxic})")
    write_tsv(merged, args.out_merged)


if __name__ == "__main__":
    main()
