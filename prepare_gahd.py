"""
Prepare additional German hate speech data for training.

Downloads two public datasets and merges with existing train_germeval.tsv:

1. GAHD (German Adversarial Hate Speech Dataset)
   Source : https://github.com/jagol/gahd
   License: CC-BY-4.0
   Size   : ~11,000 examples (German adversarial hate speech)
   Labels : label=1 (hate), label=0 (not hate)

2. GermEval 2021 Shared Task (Toxic Comment Detection)
   Source : https://github.com/germeval2021toxic/SharedTask
   Labels : Sub1_Toxic=1 (toxic), Sub1_Toxic=0 (not toxic)
   Size   : ~4,188 Facebook comments

Output:
  train_germeval_gahd.tsv  — train_germeval.tsv + GAHD + GermEval 2021
"""
from __future__ import annotations

import csv
import io
import urllib.request
from pathlib import Path


GAHD_URL = (
    "https://raw.githubusercontent.com/jagol/gahd/main/gahd.csv"
)
GERMEVAL2021_URL = (
    "https://raw.githubusercontent.com/germeval2021toxic/SharedTask/main/"
    "Data%20Sets/GermEval21_TrainData.csv"
)


def fetch_text(url: str) -> str:
    print(f"  Downloading: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


def load_gahd(use_splits=("train", "dev", "test")) -> list[dict]:
    """Load GAHD CSV. Columns: gahd_id, text, label, round, split, contrastive_gahd_id"""
    content = fetch_text(GAHD_URL)
    reader = csv.DictReader(io.StringIO(content))
    rows = []
    for i, r in enumerate(reader):
        if r.get("split", "train") not in use_splits:
            continue
        text  = r["text"].strip()
        label = int(r["label"])
        rows.append({"id": f"ger_gahd_{i}", "text": text, "label": label})
    toxic = sum(r["label"] for r in rows)
    print(f"  GAHD: {len(rows)} rows  (toxic={toxic}, clean={len(rows)-toxic})")
    return rows


def load_germeval2021() -> list[dict]:
    """Load GermEval 2021. Columns: comment_id, comment_text, Sub1_Toxic, ..."""
    content = fetch_text(GERMEVAL2021_URL)
    # CSV may use semicolons or commas; try comma first
    sample = content[:2000]
    delimiter = ";" if sample.count(";") > sample.count(",") else ","
    reader = csv.DictReader(io.StringIO(content), delimiter=delimiter)
    rows = []
    for i, r in enumerate(reader):
        text  = r.get("comment_text", "").strip()
        toxic = r.get("Sub1_Toxic", "0").strip()
        label = 1 if toxic == "1" else 0
        if not text:
            continue
        rows.append({"id": f"ger_germeval2021_{i}", "text": text, "label": label})
    toxic = sum(r["label"] for r in rows)
    print(f"  GermEval 2021: {len(rows)} rows  (toxic={toxic}, clean={len(rows)-toxic})")
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
    ap.add_argument("--base_tsv", default="train_germeval.tsv",
                    help="Existing merged train+GermEval2018 file")
    ap.add_argument("--out_tsv",  default="train_germeval_gahd.tsv")
    ap.add_argument("--no_gahd",          action="store_true")
    ap.add_argument("--no_germeval2021",   action="store_true")
    args = ap.parse_args()

    base = read_tsv(args.base_tsv)
    print(f"Base ({args.base_tsv}): {len(base)} rows")

    extra: list[dict] = []

    if not args.no_gahd:
        print("\nLoading GAHD ...")
        try:
            extra.extend(load_gahd())
        except Exception as e:
            print(f"  WARNING: GAHD download failed: {e}")

    if not args.no_germeval2021:
        print("\nLoading GermEval 2021 ...")
        try:
            extra.extend(load_germeval2021())
        except Exception as e:
            print(f"  WARNING: GermEval 2021 download failed: {e}")

    merged = base + extra
    toxic  = sum(1 for r in merged if int(r["label"]) == 1)
    print(f"\nMerged: {len(merged)} total  (toxic={toxic}, clean={len(merged)-toxic})")
    write_tsv(merged, args.out_tsv)


if __name__ == "__main__":
    main()
