"""
E2: Per-language threshold tuning on top of a trained XLM-R model.

Strategy:
  1. Load model from --model_dir (e.g. artifacts/xlmr-e1)
  2. Run inference on dev set, collect per-example softmax probabilities
  3. For each language independently, grid-search threshold in [t_min, t_max]
     that maximises F1-macro on dev
  4. Apply best thresholds to test set, write predictions
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def write_predictions_tsv(ids, preds, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "label"])
        for eid, pred in zip(ids, preds):
            w.writerow([eid, int(pred)])


def get_probabilities(model, tokenizer, examples, batch_size, max_length, device):
    """Run inference, return (ids, langs, p_toxic) for each example."""
    import torch
    from torch.utils.data import DataLoader
    from toxicity.data import infer_lang

    class SimpleDS:
        def __init__(self, exs):
            self.exs = exs
        def __len__(self):
            return len(self.exs)
        def __getitem__(self, i):
            return {"id": self.exs[i].id, "text": self.exs[i].text}

    def collate(batch):
        texts = [b["text"] for b in batch]
        ids   = [b["id"]   for b in batch]
        langs = [infer_lang(b["id"]) for b in batch]
        enc = tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        enc["ids"]   = ids
        enc["langs"] = langs
        return enc

    loader = DataLoader(
        SimpleDS(examples), batch_size=batch_size,
        shuffle=False, num_workers=0, collate_fn=collate,
    )
    model.eval()
    all_ids, all_langs, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            ids_b   = batch.pop("ids")
            langs_b = batch.pop("langs")
            batch = {k: v.to(device) if hasattr(v, "to") else v
                     for k, v in batch.items()}
            logits = model(**batch).logits          # (B, 2)
            probs  = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_ids.extend(ids_b)
            all_langs.extend(langs_b)
            all_probs.extend(probs.tolist())

    return all_ids, all_langs, all_probs


def search_threshold(y_true, probs, thresholds):
    """Grid search: return (best_t, best_f1_macro)."""
    from sklearn.metrics import f1_score
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        preds = [1 if p >= t else 0 for p in probs]
        if len(set(preds)) < 2:          # all-same → skip
            continue
        f1 = f1_score(y_true, preds, average="macro")
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def main():
    ap = argparse.ArgumentParser(
        description="E2: Per-language threshold tuning for XLM-R.")
    ap.add_argument("--model_dir",  default="artifacts/xlmr-e1",
                    help="Saved model/tokenizer directory (E1 checkpoint)")
    ap.add_argument("--dev_tsv",    default="dev.tsv")
    ap.add_argument("--test_tsv",   default="test.tsv")
    ap.add_argument("--out_tsv",    default="artifacts/xlmr-e2_test.tsv")
    ap.add_argument("--max_length", type=int,   default=192)
    ap.add_argument("--batch_size", type=int,   default=64)
    ap.add_argument("--step",       type=float, default=0.01,
                    help="Threshold grid step size")
    ap.add_argument("--t_min",      type=float, default=0.25)
    ap.add_argument("--t_max",      type=float, default=0.70)
    ap.add_argument("--device",     default="auto")
    args = ap.parse_args()

    # ── device ──────────────────────────────────────────────────────────
    import torch
    from toxicity.utils import pick_device
    di = pick_device(None if args.device == "auto" else args.device)
    device = torch.device(di.device)
    print(f"Device: {di.device}  backend={di.backend}")

    # ── load model ───────────────────────────────────────────────────────
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    print(f"Loading model from: {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device).eval()

    # ── load data ────────────────────────────────────────────────────────
    from toxicity.data import read_tsv
    dev_examples  = read_tsv(args.dev_tsv,  has_label=True)
    test_examples = read_tsv(args.test_tsv, has_label=False)

    # ── dev inference ────────────────────────────────────────────────────
    print("Running inference on dev set ...")
    dev_ids, dev_langs, dev_probs = get_probabilities(
        model, tokenizer, dev_examples,
        args.batch_size, args.max_length, device,
    )
    dev_labels = {ex.id: ex.label for ex in dev_examples}

    # group by language
    lang_probs  = defaultdict(list)
    lang_labels = defaultdict(list)
    for eid, lang, prob in zip(dev_ids, dev_langs, dev_probs):
        lang_probs[lang].append(prob)
        lang_labels[lang].append(dev_labels[eid])

    # ── threshold search ─────────────────────────────────────────────────
    thresholds = np.arange(args.t_min, args.t_max + 1e-9, args.step).tolist()
    best_thresholds: dict[str, float] = {}

    from sklearn.metrics import f1_score
    print(f"\nGrid search: t in [{args.t_min:.2f}, {args.t_max:.2f}]  step={args.step}")
    print("-" * 65)
    for lang in sorted(lang_probs):
        t, f1_tuned = search_threshold(
            lang_labels[lang], lang_probs[lang], thresholds)
        best_thresholds[lang] = t

        preds_default = [1 if p >= 0.5 else 0 for p in lang_probs[lang]]
        f1_default = f1_score(lang_labels[lang], preds_default, average="macro")
        print(f"  {lang:>4}:  t=0.50 -> f1={f1_default:.4f}  |  "
              f"t={t:.2f}  -> f1={f1_tuned:.4f}   (delta={f1_tuned - f1_default:+.4f})")
    print("-" * 65)
    print(f"Best thresholds: {best_thresholds}")

    # ── test inference + apply thresholds ────────────────────────────────
    print("\nRunning inference on test set ...")
    test_ids, test_langs, test_probs = get_probabilities(
        model, tokenizer, test_examples,
        args.batch_size, args.max_length, device,
    )
    test_preds = [
        1 if prob >= best_thresholds.get(lang, 0.5) else 0
        for lang, prob in zip(test_langs, test_probs)
    ]
    write_predictions_tsv(test_ids, test_preds, args.out_tsv)
    print(f"Wrote {len(test_preds)} test predictions -> {args.out_tsv}")

    # ── dev report with tuned thresholds ─────────────────────────────────
    from toxicity.metrics import print_report
    tuned_dev_preds = [
        1 if prob >= best_thresholds.get(lang, 0.5) else 0
        for lang, prob in zip(dev_langs, dev_probs)
    ]
    dev_true = [dev_labels[eid] for eid in dev_ids]
    print_report(
        title="DEV (xlmr-e2  tuned thresholds)",
        y_true=dev_true, y_pred=tuned_dev_preds, langs=dev_langs,
    )


if __name__ == "__main__":
    main()
