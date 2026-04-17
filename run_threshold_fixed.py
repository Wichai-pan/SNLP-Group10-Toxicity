"""
E2: Per-language threshold tuning on top of a trained XLM-R model.
FIXED: outputs 'predicted' column (Codabench compatible) and handles all 12,791 test IDs.

Strategy:
  1. Load model from --model_dir
  2. Run inference on dev set, collect per-example softmax probabilities
  3. Per-language grid-search threshold in [t_min, t_max] maximising F1-macro on dev
  4. Apply best thresholds to ALL 12,791 test IDs, write predictions
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

VALID_ID = re.compile(r'^(eng|ger|fin)_(?:test|dev|train)_\d+')


def write_predictions_tsv(ids, preds, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "predicted"])   # Codabench expects 'predicted'
        for eid, pred in zip(ids, preds):
            w.writerow([eid, int(pred)])


def read_all_ids_linewise(path):
    """Read all valid IDs via line-by-line parsing (handles embedded newlines in text)."""
    ids = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n\r').split('\t')
            if parts and VALID_ID.match(parts[0]):
                ids.append(parts[0])
    return ids


def get_probabilities(model, tokenizer, examples, batch_size, max_length, device):
    """Run inference, return (ids, langs, p_toxic) for each example."""
    import torch
    from torch.utils.data import DataLoader
    from toxicity.data import infer_lang

    class SimpleDS:
        def __init__(self, exs): self.exs = exs
        def __len__(self): return len(self.exs)
        def __getitem__(self, i): return {"id": self.exs[i].id, "text": self.exs[i].text}

    def collate(batch):
        texts = [b["text"] for b in batch]
        ids   = [b["id"]   for b in batch]
        langs = [infer_lang(b["id"]) for b in batch]
        enc = tokenizer(texts, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        enc["ids"]   = ids
        enc["langs"] = langs
        return enc

    loader = DataLoader(SimpleDS(examples), batch_size=batch_size,
                        shuffle=False, num_workers=0, collate_fn=collate)
    model.eval()
    all_ids, all_langs, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            ids_b   = batch.pop("ids")
            langs_b = batch.pop("langs")
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            logits = model(**batch).logits
            probs  = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_ids.extend(ids_b)
            all_langs.extend(langs_b)
            all_probs.extend(probs.tolist())

    return all_ids, all_langs, all_probs


def search_threshold(y_true, probs, thresholds):
    from sklearn.metrics import f1_score
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        preds = [1 if p >= t else 0 for p in probs]
        if len(set(preds)) < 2:
            continue
        f1 = f1_score(y_true, preds, average="macro")
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir",  default="artifacts/xlmr-e1")
    ap.add_argument("--dev_tsv",    default="dev.tsv")
    ap.add_argument("--test_tsv",   default="test.tsv")
    ap.add_argument("--out_tsv",    default="artifacts/xlmr-e2_test.tsv")
    ap.add_argument("--max_length", type=int,   default=192)
    ap.add_argument("--batch_size", type=int,   default=64)
    ap.add_argument("--step",       type=float, default=0.01)
    ap.add_argument("--t_min",      type=float, default=0.10)
    ap.add_argument("--t_max",      type=float, default=0.90)
    ap.add_argument("--device",     default="auto")
    args = ap.parse_args()

    import torch
    from toxicity.utils import pick_device
    di = pick_device(None if args.device == "auto" else args.device)
    device = torch.device(di.device)
    print(f"Device: {di.device}  backend={di.backend}")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    print(f"Loading model from: {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device).eval()

    from toxicity.data import read_tsv
    dev_examples  = read_tsv(args.dev_tsv,  has_label=True)
    test_examples = read_tsv(args.test_tsv, has_label=False)

    print("Running inference on dev ...")
    dev_ids, dev_langs, dev_probs = get_probabilities(
        model, tokenizer, dev_examples, args.batch_size, args.max_length, device)
    dev_labels = {ex.id: ex.label for ex in dev_examples}

    lang_probs  = defaultdict(list)
    lang_labels = defaultdict(list)
    for eid, lang, prob in zip(dev_ids, dev_langs, dev_probs):
        lang_probs[lang].append(prob)
        lang_labels[lang].append(dev_labels[eid])

    thresholds = np.arange(args.t_min, args.t_max + 1e-9, args.step).tolist()
    best_thresholds: dict[str, float] = {}

    from sklearn.metrics import f1_score
    print(f"\nGrid search: t in [{args.t_min:.2f}, {args.t_max:.2f}]  step={args.step}")
    print("-" * 65)
    for lang in sorted(lang_probs):
        t, f1_tuned = search_threshold(lang_labels[lang], lang_probs[lang], thresholds)
        best_thresholds[lang] = t
        preds_default = [1 if p >= 0.5 else 0 for p in lang_probs[lang]]
        f1_default = f1_score(lang_labels[lang], preds_default, average="macro")
        print(f"  {lang:>4}:  t=0.50 -> f1={f1_default:.4f}  |  "
              f"t={t:.2f}  -> f1={f1_tuned:.4f}   (delta={f1_tuned - f1_default:+.4f})")
    print("-" * 65)
    print(f"Best thresholds: {best_thresholds}")

    # Test inference
    print("\nRunning inference on test set ...")
    test_ids, test_langs, test_probs = get_probabilities(
        model, tokenizer, test_examples, args.batch_size, args.max_length, device)

    # Use all 12,791 IDs (handles embedded newline rows)
    all_test_ids = read_all_ids_linewise(args.test_tsv)
    id_to_prob = {eid: p for eid, p in zip(test_ids, test_probs)}
    id_to_lang = {eid: l for eid, l in zip(test_ids, test_langs)}

    final_ids, final_preds = [], []
    for eid in all_test_ids:
        lang = id_to_lang.get(eid, eid.split('_')[0])
        prob = id_to_prob.get(eid, 0.0)
        pred = 1 if prob >= best_thresholds.get(lang, 0.5) else 0
        final_ids.append(eid)
        final_preds.append(pred)

    write_predictions_tsv(final_ids, final_preds, args.out_tsv)
    print(f"Wrote {len(final_preds)} test predictions -> {args.out_tsv}")

    # Dev report
    from toxicity.metrics import print_report
    tuned_dev_preds = [
        1 if p >= best_thresholds.get(l, 0.5) else 0
        for l, p in zip(dev_langs, dev_probs)
    ]
    dev_true = [dev_labels[eid] for eid in dev_ids]
    print_report(
        title=f"DEV ({Path(args.model_dir).name} tuned thresholds)",
        y_true=dev_true, y_pred=tuned_dev_preds, langs=dev_langs,
    )


if __name__ == "__main__":
    main()
