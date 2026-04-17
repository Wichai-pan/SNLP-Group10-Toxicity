"""
Experiment 4: Prepare GermEval 2018 data for training.

Downloads GermEval 2018 Task 1 (coarse offensive language detection)
from GitHub and converts it to our TSV format, then merges with the
original English training set.

GermEval label mapping:
  OFFENSE -> 1 (toxic)
  OTHER   -> 0 (clean)

Output:
  germeval_train.tsv   - GermEval training data only (German)
  train_germeval.tsv   - Original English train + GermEval German
"""
from __future__ import annotations

import csv
import urllib.request
from pathlib import Path


GERMEVAL_URLS = {
    "train": "https://raw.githubusercontent.com/uds-lsv/GermEval-2018-Data/master/germeval2018.training.txt",
    "test":  "https://raw.githubusercontent.com/uds-lsv/GermEval-2018-Data/master/germeval2018.test.txt",
}


def download_germeval(url: str) -> list[dict]:
    """Download and parse GermEval TSV (text\\tcoarse\\tfine)."""
    print(f"Downloading: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read().decode("utf-8", errors="replace")

    rows = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        text = parts[0].strip()
        coarse = parts[1].strip().upper()
        label = 1 if coarse == "OFFENSE" else 0
        rows.append({"text": text, "label": label})

    print(f"  Parsed {len(rows)} examples  (toxic={sum(r['label'] for r in rows)}, "
          f"clean={sum(1-r['label'] for r in rows)})")
    return rows


def read_tsv(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append({"id": row["id"], "text": row["text"], "label": row["label"]})
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
    ap.add_argument("--train_tsv",  default="train.tsv")
    ap.add_argument("--out_dir",    default=".")
    ap.add_argument("--splits",     default="train",
                    help="Comma-separated GermEval splits to include: train,test")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    splits = [s.strip() for s in args.splits.split(",")]

    # Download GermEval
    all_ger_rows = []
    for split in splits:
        if split not in GERMEVAL_URLS:
            print(f"Unknown split: {split}, skipping")
            continue
        rows = download_germeval(GERMEVAL_URLS[split])
        for i, r in enumerate(rows):
            r["id"] = f"ger_germeval_{split}_{i}"
        all_ger_rows.extend(rows)

    if not all_ger_rows:
        print("No GermEval data downloaded, exiting.")
        return

    # Save standalone GermEval file
    germeval_path = str(out_dir / "germeval_train.tsv")
    write_tsv(all_ger_rows, germeval_path)

    # Load original English training data
    eng_rows = read_tsv(args.train_tsv)
    print(f"Original train.tsv: {len(eng_rows)} English examples")

    # Merge: original English + GermEval German
    merged = eng_rows + all_ger_rows
    label_counts = {0: sum(1 for r in merged if str(r["label"]) == "0"),
                    1: sum(1 for r in merged if str(r["label"]) == "1")}
    print(f"Merged: {len(merged)} total  (toxic={label_counts[1]}, clean={label_counts[0]})")

    merged_path = str(out_dir / "train_germeval.tsv")
    write_tsv(merged, merged_path)


if __name__ == "__main__":
    main()
