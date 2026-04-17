"""
E3: Translate-Train data augmentation.

Translates the English training set to German and Finnish using
Helsinki-NLP MarianMT models, writes translated TSVs incrementally,
and builds a merged multilingual training file.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from transformers import MarianMTModel, MarianTokenizer


def read_train_tsv(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append(
                {
                    "id": row["id"],
                    "text": row["text"],
                    "label": row["label"],
                }
            )
    return rows


def count_tsv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader, None)
        return sum(1 for _ in reader)


def open_tsv_writer(path: Path, append: bool) -> tuple[object, csv.writer]:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    handle = open(path, mode, encoding="utf-8", newline="")
    writer = csv.writer(handle, delimiter="\t")
    if not append:
        writer.writerow(["id", "text", "label"])
        handle.flush()
    return handle, writer


def translate_batch(
    texts: list[str],
    tokenizer: MarianTokenizer,
    model: MarianMTModel,
    device: torch.device,
    max_length: int,
) -> list[str]:
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)
    with torch.no_grad():
        translated = model.generate(
            **inputs,
            num_beams=4,
            max_length=max_length,
            early_stopping=True,
        )
    return tokenizer.batch_decode(translated, skip_special_tokens=True)


def translate_to_tsv(
    rows: list[dict],
    model_name: str,
    out_path: Path,
    id_prefix: str,
    device: torch.device,
    batch_size: int,
    max_length: int,
    log_every: int,
    resume: bool,
    lang_label: str,
) -> int:
    existing = count_tsv_rows(out_path) if resume else 0
    if existing > len(rows):
        raise ValueError(
            f"{out_path} has {existing} rows, but source only has {len(rows)} rows."
        )
    if existing == len(rows):
        print(
            f"[{lang_label}] Output already complete: {out_path} ({existing} rows)",
            flush=True,
        )
        return existing

    print(f"\n[{lang_label}] Loading translation model: {model_name}", flush=True)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
    model.eval()

    append = existing > 0
    handle, writer = open_tsv_writer(out_path, append=append)
    start_time = time.time()
    print(
        f"[{lang_label}] Starting at row {existing}/{len(rows)} -> {out_path}",
        flush=True,
    )

    try:
        for batch_index, start in enumerate(range(existing, len(rows), batch_size), start=1):
            batch = rows[start : start + batch_size]
            texts = [row["text"] for row in batch]
            try:
                translations = translate_batch(
                    texts=texts,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    max_length=max_length,
                )
            except Exception as exc:
                print(
                    f"[{lang_label}] Warning: batch starting at {start} failed ({exc}); "
                    "falling back to original texts",
                    flush=True,
                )
                translations = texts

            for source_row, translated_text in zip(batch, translations):
                writer.writerow([f"{id_prefix}_{source_row['id']}", translated_text, source_row["label"]])
            handle.flush()

            done = start + len(batch)
            if batch_index == 1 or batch_index % log_every == 0 or done == len(rows):
                elapsed = time.time() - start_time
                processed = done - existing
                rate = processed / elapsed if elapsed > 0 else 0.0
                remaining = len(rows) - done
                eta_minutes = (remaining / rate / 60.0) if rate > 0 else 0.0
                print(
                    f"[{lang_label}] {done}/{len(rows)} ({100.0 * done / len(rows):.1f}%) "
                    f"| {rate:.1f} ex/s | ETA {eta_minutes:.1f} min",
                    flush=True,
                )
    finally:
        handle.close()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    total_elapsed = time.time() - start_time
    print(
        f"[{lang_label}] Finished {len(rows)} rows in {total_elapsed / 60.0:.1f} min",
        flush=True,
    )
    return len(rows)


def write_merged_tsv(orig_rows: list[dict], translated_paths: list[Path], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["id", "text", "label"])

        for row in orig_rows:
            writer.writerow([row["id"], row["text"], row["label"]])
            total += 1

        for translated_path in translated_paths:
            if not translated_path.exists():
                continue
            with open(translated_path, encoding="utf-8", newline="") as translated_handle:
                reader = csv.DictReader(translated_handle, delimiter="\t")
                for row in reader:
                    writer.writerow([row["id"], row["text"], row["label"]])
                    total += 1
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate-Train data augmentation")
    parser.add_argument("--train_tsv", default="train.tsv")
    parser.add_argument("--out_dir", default=".")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--langs", default="de,fi")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}", flush=True)
    print(f"Reading {args.train_tsv}", flush=True)
    orig_rows = read_train_tsv(args.train_tsv)
    if args.limit is not None:
        orig_rows = orig_rows[: args.limit]
        print(f"Using limited subset: {len(orig_rows)} rows", flush=True)
    else:
        print(f"Loaded {len(orig_rows)} rows", flush=True)

    out_dir = Path(args.out_dir)
    target_langs = [lang.strip() for lang in args.langs.split(",") if lang.strip()]
    model_names = {
        "de": "Helsinki-NLP/opus-mt-en-de",
        "fi": "Helsinki-NLP/opus-mt-en-fi",
    }
    id_prefixes = {
        "de": "ger",
        "fi": "fin",
    }

    translated_paths: list[Path] = []

    for lang in target_langs:
        if lang not in model_names:
            print(f"Skipping unknown language: {lang}", flush=True)
            continue

        out_path = out_dir / f"train_{lang}.tsv"
        translated_paths.append(out_path)

        if args.skip_existing and count_tsv_rows(out_path) == len(orig_rows):
            print(f"[{lang.upper()}] Skipping completed file: {out_path}", flush=True)
            continue

        translate_to_tsv(
            rows=orig_rows,
            model_name=model_names[lang],
            out_path=out_path,
            id_prefix=id_prefixes[lang],
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            log_every=args.log_every,
            resume=True,
            lang_label=lang.upper(),
        )

    merged_path = out_dir / "train_aug.tsv"
    total_rows = write_merged_tsv(orig_rows=orig_rows, translated_paths=translated_paths, out_path=merged_path)
    print(f"Wrote merged training set: {merged_path} ({total_rows} rows)", flush=True)


if __name__ == "__main__":
    main()
