# Multilingual Toxicity Detection — SNLP Group 10 (Aalto 2026)

Cross-lingual toxicity classification: **English-only training → English / German / Finnish evaluation**.  
Course competition for ELEC-E5550 Statistical Natural Language Processing, Aalto University.

> **Data licence**: The dataset is provided under a course-specific agreement. Do **not** redistribute `train.tsv`, `dev.tsv`, or `test.tsv`.

---

## Task

Binary classification of short social-media texts as **toxic (1) / non-toxic (0)**.

| Split | Size | Languages |
|-------|------|-----------|
| Train | 98 k | English only |
| Dev   | 13.2 k | English (83 %), German (15 %), Finnish (2 %) |
| Test  | 12.8 k | English (47 %), German (47 %), Finnish (6 %) |

Language is inferred from the `id` prefix: `eng_*`, `ger_*`, `fin_*`.

---

## Results Summary

All scores are **macro-F1** on the dev set. Full results in [`results/dev_results.csv`](results/dev_results.csv).

| ID | Model | Training Data | ENG | GER | FIN | Overall |
|----|-------|---------------|-----|-----|-----|---------|
| E00 | TF-IDF + LogReg | train.tsv | 0.905 | 0.505 | 0.233 | 0.861 |
| E01 | XLM-R base | train.tsv | 0.943 | 0.678 | 0.748 | 0.910 |
| E02 | E01 + threshold | train.tsv | 0.944 | 0.688 | 0.759 | 0.910 |
| E03 | Twitter-XLM-R base | train.tsv | 0.943 | 0.690 | 0.533 | 0.907 |
| E05 | XLM-R base + translate-train | train_aug.tsv | 0.939 | 0.703 | 0.777 | 0.909 |
| E06 | XLM-R base + GermEval | train_germeval.tsv | 0.941 | 0.721 | 0.715 | 0.908 |
| E07 | XLM-R base + back-translation | train_backtrans.tsv | 0.936 | 0.685 | 0.743 | 0.903 |
| L01 | XLM-R large | train.tsv | 0.946 | 0.720 | 0.750 | 0.916 |
| L02 | XLM-R large + translate-train | train_aug.tsv | 0.945 | 0.706 | **0.797** | 0.915 |
| L03 | XLM-R large + GermEval ⭐ | train_germeval.tsv | 0.948 | **0.751** | 0.790 | 0.919 |
| L03b | XLM-R large + GermEval (seed 123) | train_germeval.tsv | 0.944 | 0.748 | 0.759 | 0.916 |
| L05 | XLM-R large + translate + GermEval | train_aug_germeval.tsv | 0.942 | 0.735 | 0.797 | 0.912 |
| L06 | XLM-R large + GermEval+GAHD | train_germeval_gahd.tsv | 0.947 | 0.738 | 0.778 | 0.917 |
| L07 | XLM-R large + translate + Suomi24 | train_aug_suomi24.tsv | 0.946 | 0.703 | 0.274 | 0.908 |
| O01 | L03 + threshold | — | 0.948 | 0.759 | 0.822 | 0.922 |
| O02 | Ensemble L02+L03 | — | 0.951 | 0.745 | 0.814 | 0.924 |
| O03 | Ensemble L02+L03+L05 | — | 0.950 | 0.755 | **0.826** | 0.925 |
| **O04** ⭐⭐ | **Ensemble L02+L03+L03b** | — | **0.951** | 0.754 | 0.812 | **0.926** |
| O05 | Ensemble L02+L06 | — | 0.951 | 0.735 | 0.813 | 0.922 |

**Key findings:**
- **GermEval 2018** (native German Twitter) is the most effective single addition for German (+0.031 over L01)
- **Translate-train** (EN→DE/FI) is most effective for Finnish (+0.047 over L01)
- **Ensembling** complementary specialists (L02+L03) outperforms any single model and combined-data training
- **Back-translation** (E07) and **Twitter-XLM-R** (E03) hurt Finnish — domain mismatch matters more than language match
- **External Finnish data** (L07, Suomi24) collapses Finnish F1 to 0.274 due to domain mismatch

---

## Repository Structure

```
.
├── run_baseline.py          # E00: TF-IDF + LogReg baseline
├── run_xlmr.py              # E01–L07: XLM-R fine-tuning (train + eval + predict)
├── run_threshold_fixed.py   # Per-language threshold tuning (grid search)
├── run_ensemble.py          # O02: 2-model ensemble (softmax averaging)
├── run_ensemble_n.py        # O03–O05: N-model ensemble
├── translate_dataset.py     # EN → DE/FI translation (Helsinki-NLP OPUS-MT)
├── back_translate.py        # DE/FI → EN back-translation
├── prepare_germeval.py      # Download & merge GermEval 2018
├── prepare_gahd.py          # Download & merge GAHD dataset
├── prepare_suomi24.py       # Download & merge Suomi24 Finnish data
├── make_aug_germeval.py     # Combine train_aug + GermEval → train_aug_germeval.tsv
├── fix_submissions.py       # Fix row count for Codabench (embedded newlines)
│
├── toxicity/                # Shared library
│   ├── data.py              # read_tsv, Example dataclass, infer_lang
│   ├── metrics.py           # compute_metrics, print_report (per-language F1)
│   └── utils.py             # seed_everything, pick_device
│
├── scripts/
│   ├── experiments/         # Slurm job scripts (one per experiment ID)
│   │   ├── e00_baseline.sh … o05_ensemble_gahd.sh
│   ├── data_prep/           # Data preparation jobs
│   │   ├── prep_translate.sh, prep_germeval.sh, prep_gahd.sh …
│   └── util/                # Environment & workflow utilities
│       ├── util_setup_puhti.sh
│       ├── util_setup_mac.sh
│       └── util_sync_to_puhti.sh
│
├── results/
│   └── dev_results.csv      # All dev-set macro-F1 results
│
├── docs/
│   └── puhti_experiment_plan.md
└── requirements.txt
```

---

## Setup

### Local (macOS / Linux)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Quick smoke test (CPU, tiny data)
python run_xlmr.py --preset fast --limit_train 512 --limit_dev 256 --epochs 1
```

### Puhti HPC (CSC Finland)

> **Before submitting jobs**, edit `#SBATCH --account=YOUR_CSC_PROJECT` in each script to your project ID.  
> Submit from the repo root so relative `--output=runs/…` paths resolve correctly.

```bash
# 1. Copy repo to your scratch space
rsync -av ./ <username>@puhti.csc.fi:/scratch/<project>/<username>/SNLP_Group_Work/

# 2. (Optional) point HF cache to fast scratch
export SNLP_HF_HOME=/scratch/<project>/<username>/hf-cache

# 3. Submit jobs from repo root
cd /scratch/<project>/<username>/SNLP_Group_Work
sbatch scripts/experiments/l03_large_germeval.sh
```

### Other HPC / Generic GPU

The scripts use CSC-specific module loading (`module load pytorch/2.5`).  
On other systems, replace those lines with your environment activation (e.g. `conda activate`, `source .venv/bin/activate`).

---

## Reproducing the Best Results (O04)

```bash
# Step 1 — Prepare augmented data (~3–4h on GPU)
sbatch scripts/data_prep/prep_translate.sh   # → train_aug.tsv
sbatch scripts/data_prep/prep_germeval.sh    # → train_germeval.tsv

# Step 2 — Train specialist models (~6–7h each on V100)
sbatch scripts/experiments/l02_large_aug.sh          # FIN specialist
sbatch scripts/experiments/l03_large_germeval.sh     # GER specialist
sbatch scripts/experiments/l03b_large_germeval_seed2.sh  # GER specialist, seed 123

# Step 3 — 3-model ensemble (no extra training, ~15 min)
sbatch scripts/experiments/o04_ensemble_seeds.sh
# → artifacts/o04-ensemble-seeds-th_test.tsv  (dev F1 = 0.926)

# Step 4 — Fix submission file for Codabench
python fix_submissions.py
```

---

## Core Scripts

### `run_xlmr.py`

Fine-tunes any HuggingFace sequence classification checkpoint.

```bash
python run_xlmr.py \
  --model_name xlm-roberta-large \
  --train_tsv  train_germeval.tsv \
  --preset     score \
  --epochs     4 \
  --gradient_checkpointing \
  --train_batch_size 4 --grad_accum_steps 8 \
  --out_dir    artifacts/my-model \
  --out_tsv    artifacts/my-model_test.tsv \
  --predict_test
```

| Flag | Default | Description |
|------|---------|-------------|
| `--preset score` | — | max_length=192, lr=1e-5, batch=32 |
| `--preset fast` | — | max_length=128, lr=2e-5, batch=16 (for quick tests) |
| `--gradient_checkpointing` | off | Required for large model on ≤40 GB GPU |
| `--predict_only` | off | Skip training; run inference from `--out_dir` |

### `run_threshold_fixed.py`

Per-language threshold grid search on dev set.

```bash
python run_threshold_fixed.py \
  --model_dir  artifacts/l03-large-germeval \
  --t_min 0.10 --t_max 0.90 --step 0.01
```

### `run_ensemble_n.py`

Averages softmax probabilities from N checkpoints, then tunes thresholds.

```bash
python run_ensemble_n.py \
  --model_dirs artifacts/l02-large-aug \
               artifacts/l03-large-germeval \
               artifacts/l03b-large-germeval \
  --out_tsv    artifacts/o04_test.tsv
```

---

## Data Pipeline

```
train.tsv (EN, 98k)
    │
    ├─ translate_dataset.py ──→ train_de.tsv + train_fi.tsv
    │                                └─ (concat) train_aug.tsv (235k)  [used by L02]
    │
    ├─ prepare_germeval.py ───→ germeval_train.tsv (8.5k DE native)
    │                                └─ (merge with train.tsv) train_germeval.tsv (107k)  [used by L03]
    │
    └─ make_aug_germeval.py ──→ train_aug_germeval.tsv (244k)  [used by L05]
```

Translation models (Helsinki-NLP, no API key needed):
- EN→DE: `Helsinki-NLP/opus-mt-en-de`
- EN→FI: `Helsinki-NLP/opus-mt-en-fi`

---

## Requirements

```
torch>=2.0
transformers>=4.38
scikit-learn>=1.3
numpy
tqdm
sentencepiece
sacremoses
```

---

## References

- **XLM-RoBERTa**: Conneau et al. (2020) — *Unsupervised Cross-lingual Representation Learning at Scale*
- **GermEval 2018**: Wiegand et al. (2018) — German offensive language shared task
- **GAHD**: Goldzycher & Schneider (2022) — German Adversarial Hate Speech Dataset
- **Helsinki-NLP OPUS-MT**: Tiedemann & Thottingal (2020)
- **Back-translation**: Sennrich et al. (2016)
