"""
Create train_aug_germeval.tsv:
  train_aug.tsv (EN + translated DE + translated FI, ~235k)
+ GermEval-only rows from train_germeval.tsv (~29k native German)
= ~264k total rows

Logic: take all rows from train_germeval.tsv whose IDs are NOT in train.tsv
(those are the native GermEval rows), append to train_aug.tsv.
"""
import csv
import os
from pathlib import Path
import sys

WORK = os.environ.get("SNLP_WORK", str(Path(".").resolve()))

def read_ids(path):
    ids = set()
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            ids.add(row['id'])
    return ids

def read_rows(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)
    return fieldnames, rows

print("Reading train.tsv IDs ...")
train_ids = read_ids(f"{WORK}/train.tsv")
print(f"  {len(train_ids)} IDs in train.tsv")

print("Reading train_germeval.tsv ...")
germ_fields, germ_rows = read_rows(f"{WORK}/train_germeval.tsv")
germeval_only = [r for r in germ_rows if r['id'] not in train_ids]
print(f"  {len(germ_rows)} total rows, {len(germeval_only)} GermEval-only rows")

print("Reading train_aug.tsv ...")
aug_fields, aug_rows = read_rows(f"{WORK}/train_aug.tsv")
print(f"  {len(aug_rows)} rows in train_aug.tsv")

out_path = f"{WORK}/train_aug_germeval.tsv"
total = len(aug_rows) + len(germeval_only)
print(f"\nWriting {total} rows to {out_path} ...")

with open(out_path, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=aug_fields, delimiter='\t')
    w.writeheader()
    for row in aug_rows:
        w.writerow({k: row.get(k, '') for k in aug_fields})
    for row in germeval_only:
        w.writerow({k: row.get(k, '') for k in aug_fields})

# Verify
with open(out_path, encoding='utf-8') as f:
    line_count = sum(1 for _ in f)
print(f"Done. {line_count} lines in output (including header).")
print(f"Expected: {total + 1}")
