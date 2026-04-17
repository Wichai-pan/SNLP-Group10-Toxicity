#!/bin/sh
# E00: TF-IDF + Logistic Regression Baseline
# Model  : TF-IDF (char+word n-grams) + sklearn LogisticRegression
# Data   : train.tsv (EN only, 98k)
# Result : overall F1=0.8605  ENG=0.9048  GER=0.5048  FIN=0.2329
# Note   : No cross-lingual transfer; fails on GER/FIN by design.
#SBATCH --job-name=e00-baseline
#SBATCH --account=YOUR_CSC_PROJECT
#SBATCH --partition=small
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=runs/%x-%j.out

set -e
WORK=$(cd "$(dirname "$0")/../.." && pwd)

. /appl/profile/zz-csc-env.sh
set -u
module load pytorch/2.5

export PYTHONUNBUFFERED=1
mkdir -p "$WORK/artifacts" "$WORK/runs"
cd "$WORK"

echo "=== E00: TF-IDF + LogReg Baseline ==="
echo "Start: $(date)"

python run_baseline.py \
    --train_tsv  train.tsv \
    --dev_tsv    dev.tsv \
    --test_tsv   test.tsv \
    --predict_test \
    --model_out  artifacts/e00-baseline.joblib \
    --out_tsv    artifacts/e00-baseline_test.tsv

echo "Done: $(date)"
