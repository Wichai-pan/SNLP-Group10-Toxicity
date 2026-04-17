"""
One-shot inference for three tasks:
  1. Save dev probabilities from L02, L03, L03b (for bootstrap CI + confusion matrix)
  2. Save per-model + ensemble predictions on dev (for Table 5 quantification)
  3. Save predictions on Multilingual HateCheck German (for HateCheck functional eval)

Outputs (under --out_dir):
  dev_probs.jsonl       one JSON per example, keys: id, lang, label, p_l02, p_l03, p_l03b
  hatecheck_de.jsonl    one JSON per case, keys: mhc_case_id, functionality, label_gold, test_case, p_l02, p_l03, p_l03b
  meta.json             tokenizer / max_length / model paths
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from toxicity.data import read_tsv, infer_lang


def collate_texts(texts, ids, langs, tokenizer, max_length):
    enc = tokenizer(texts, padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt")
    enc["_ids"] = ids
    enc["_langs"] = langs
    return enc


def infer_probs(model, tokenizer, texts, ids, langs, batch_size, max_length, device):
    class DS:
        def __init__(self, t, i, l):
            self.t, self.i, self.l = t, i, l
        def __len__(self): return len(self.t)
        def __getitem__(self, k): return (self.t[k], self.i[k], self.l[k])

    def collate(batch):
        ts, ids_b, langs_b = zip(*batch)
        enc = tokenizer(list(ts), padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        enc["_ids"] = list(ids_b)
        enc["_langs"] = list(langs_b)
        return enc

    loader = DataLoader(DS(texts, ids, langs), batch_size=batch_size,
                        shuffle=False, num_workers=0, collate_fn=collate)
    model.eval()
    out_ids, out_langs, out_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids_b = batch.pop("_ids")
            langs_b = batch.pop("_langs")
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            out_ids.extend(ids_b)
            out_langs.extend(langs_b)
            out_probs.extend(probs.tolist())
    return out_ids, out_langs, out_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dirs", nargs=3, required=True,
                    help="Paths for L02, L03, L03b (in that order)")
    ap.add_argument("--dev_tsv", default="dev.tsv")
    ap.add_argument("--out_dir", default="artifacts/analysis")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=192)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Prepare dev data
    dev_examples = read_tsv(args.dev_tsv, has_label=True)
    dev_texts = [e.text for e in dev_examples]
    dev_ids = [e.id for e in dev_examples]
    dev_langs = [e.lang for e in dev_examples]
    dev_labels = {e.id: e.label for e in dev_examples}
    print(f"Dev examples: {len(dev_examples)}")

    # Prepare HateCheck German
    from datasets import load_dataset
    hc = load_dataset("Paul/hatecheck-german")["test"]
    hc_ids = [f"hcde-{i}" for i in range(len(hc))]
    hc_texts = list(hc["test_case"])
    hc_langs = ["ger"] * len(hc)
    hc_gold = list(hc["label_gold"])
    hc_func = list(hc["functionality"])
    hc_case_id = list(hc["mhc_case_id"])
    print(f"HateCheck DE cases: {len(hc_texts)}")

    model_names = ["l02", "l03", "l03b"]
    dev_probs_by_model = {}
    hc_probs_by_model = {}

    for name, mdir in zip(model_names, args.model_dirs):
        print(f"\n=== Loading {name}: {mdir} ===")
        tok = AutoTokenizer.from_pretrained(mdir, use_fast=True)
        mod = AutoModelForSequenceClassification.from_pretrained(mdir).to(device).eval()

        print("  Inference on dev...")
        ids, langs, probs = infer_probs(mod, tok, dev_texts, dev_ids, dev_langs,
                                         args.batch_size, args.max_length, device)
        dev_probs_by_model[name] = {i: p for i, p in zip(ids, probs)}

        print("  Inference on HateCheck DE...")
        ids_hc, _, probs_hc = infer_probs(mod, tok, hc_texts, hc_ids, hc_langs,
                                           args.batch_size, args.max_length, device)
        hc_probs_by_model[name] = {i: p for i, p in zip(ids_hc, probs_hc)}

        del mod
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Write dev_probs.jsonl
    dev_path = out_dir / "dev_probs.jsonl"
    with dev_path.open("w", encoding="utf-8") as f:
        for ex in dev_examples:
            row = {
                "id": ex.id,
                "lang": ex.lang,
                "label": int(ex.label),
                "p_l02": dev_probs_by_model["l02"][ex.id],
                "p_l03": dev_probs_by_model["l03"][ex.id],
                "p_l03b": dev_probs_by_model["l03b"][ex.id],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote {dev_path}")

    # Write hatecheck_de.jsonl
    hc_path = out_dir / "hatecheck_de.jsonl"
    with hc_path.open("w", encoding="utf-8") as f:
        for idx, cid in enumerate(hc_ids):
            row = {
                "mhc_case_id": hc_case_id[idx],
                "functionality": hc_func[idx],
                "label_gold": hc_gold[idx],
                "test_case": hc_texts[idx],
                "p_l02": hc_probs_by_model["l02"][cid],
                "p_l03": hc_probs_by_model["l03"][cid],
                "p_l03b": hc_probs_by_model["l03b"][cid],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {hc_path}")

    # Meta
    meta = {
        "models": {n: m for n, m in zip(model_names, args.model_dirs)},
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "n_dev": len(dev_examples),
        "n_hatecheck_de": len(hc_texts),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()
