"""
Experiment 5: Back-translation data augmentation.

Pipeline:
  train.tsv (EN) -> translate to DE -> translate back to EN (paraphrase)
  train.tsv (EN) -> translate to FI -> translate back to EN (paraphrase)

The back-translated English paraphrases are added to the original training
set, giving the model diverse surface forms for the same toxic concepts.

Uses:
  Forward:  pre-existing train_de.tsv, train_fi.tsv  (from E3 translate step)
  Backward: Helsinki-NLP/opus-mt-de-en, opus-mt-fi-en

Output:
  train_backtrans_de.tsv   - DE->EN back-translations
  train_backtrans_fi.tsv   - FI->EN back-translations
  train_backtrans.tsv      - original EN + DE->EN + FI->EN
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from transformers import MarianMTModel, MarianTokenizer


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


def translate_all(rows: list[dict], model_name: str, device: torch.device,
                  batch_size: int, max_length: int, lang_label: str,
                  id_prefix: str) -> list[dict]:
    print(f"\nLoading back-translation model: {model_name}")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
    model.eval()

    translated = []
    n = len(rows)
    t0 = time.time()

    for i in range(0, n, batch_size):
        batch = rows[i: i + batch_size]
        texts = [r["text"] for r in batch]
        try:
            inputs = tokenizer(texts, return_tensors="pt", padding=True,
                               truncation=True, max_length=max_length).to(device)
            with torch.no_grad():
                out = model.generate(**inputs, num_beams=4, max_length=max_length,
                                     early_stopping=True)
            translations = tokenizer.batch_decode(out, skip_special_tokens=True)
        except Exception as e:
            print(f"  Warning: batch {i // batch_size} failed ({e}), using original texts")
            translations = texts

        for orig, trans in zip(batch, translations):
            translated.append({
                "id":    f"{id_prefix}_{orig['id']}",
                "text":  trans,
                "label": orig["label"],
            })

        if (i // batch_size) % 50 == 0:
            elapsed = time.time() - t0
            done = i + len(batch)
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n - done) / rate if rate > 0 else 0
            print(f"  [{lang_label}] {done}/{n} ({100*done/n:.1f}%)"
                  f"  {rate:.0f} ex/s  ETA {eta/60:.1f} min")

    elapsed = time.time() - t0
    print(f"  [{lang_label}] Done. {n} examples in {elapsed/60:.1f} min")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return translated


def main():
    ap = argparse.ArgumentParser(description="Back-translation augmentation")
    ap.add_argument("--train_tsv",     default="train.tsv")
    ap.add_argument("--train_de_tsv",  default="train_de.tsv",
                    help="Pre-existing EN->DE translations (from E3)")
    ap.add_argument("--train_fi_tsv",  default="train_fi.tsv",
                    help="Pre-existing EN->FI translations (from E3)")
    ap.add_argument("--out_dir",       default=".")
    ap.add_argument("--batch_size",    type=int, default=256)
    ap.add_argument("--max_length",    type=int, default=192)
    ap.add_argument("--device",        default="auto")
    ap.add_argument("--langs",         default="de,fi",
                    help="Which back-translation directions to run")
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    target_langs = [l.strip() for l in args.langs.split(",") if l.strip()]

    BACK_MODELS = {
        "de": ("Helsinki-NLP/opus-mt-de-en", args.train_de_tsv, "BT_DE", "bt_de"),
        "fi": ("Helsinki-NLP/opus-mt-fi-en", args.train_fi_tsv, "BT_FI", "bt_fi"),
    }

    # Load original English training data
    print(f"Reading {args.train_tsv} ...")
    orig_rows = read_tsv(args.train_tsv)
    print(f"  {len(orig_rows)} original English examples")

    back_translated: dict[str, list[dict]] = {}

    for lang in target_langs:
        if lang not in BACK_MODELS:
            print(f"Unknown lang: {lang}, skipping")
            continue
        model_name, src_tsv, label, id_prefix = BACK_MODELS[lang]
        out_path = out_dir / f"train_backtrans_{lang}.tsv"

        if args.skip_existing and out_path.exists():
            print(f"Skipping {lang} back-translation (output exists: {out_path})")
            back_translated[lang] = read_tsv(str(out_path))
            continue

        if not Path(src_tsv).exists():
            print(f"Source TSV not found: {src_tsv}, skipping {lang}")
            continue

        print(f"Reading forward-translated {src_tsv} ...")
        fwd_rows = read_tsv(src_tsv)
        print(f"  {len(fwd_rows)} rows")

        bt_rows = translate_all(
            fwd_rows, model_name=model_name, device=device,
            batch_size=args.batch_size, max_length=args.max_length,
            lang_label=label, id_prefix=id_prefix,
        )
        write_tsv(bt_rows, str(out_path))
        print(f"Wrote {len(bt_rows)} rows -> {out_path}")
        back_translated[lang] = bt_rows

    # Merge: original + all back-translated
    merged = list(orig_rows)
    for lang in target_langs:
        if lang in back_translated:
            merged.extend(back_translated[lang])

    aug_path = out_dir / "train_backtrans.tsv"
    write_tsv(merged, str(aug_path))
    print(f"\nWrote train_backtrans.tsv: {len(merged)} rows "
          f"({len(orig_rows)} orig + "
          f"{len(merged)-len(orig_rows)} back-translated) -> {aug_path}")


if __name__ == "__main__":
    main()
