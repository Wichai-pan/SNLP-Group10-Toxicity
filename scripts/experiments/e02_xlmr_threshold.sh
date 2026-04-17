#!/bin/sh
# E02: E01 + Per-language Threshold Tuning
# Model  : E01 checkpoint (xlm-roberta-base)
# Data   : dev.tsv for grid search, test.tsv for output
# Result : overall F1=0.9086 (+0.0004 over E01)
# Note   : Grid search t in [0.10, 0.90] per language independently.
#          Prerequisite: e01_xlmr_base.sh must have completed.
#SBATCH --job-name=e02-threshold
#SBATCH --account=YOUR_CSC_PROJECT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=runs/%x-%j.out

set -e
WORK=$(cd "$(dirname "$0")/../.." && pwd)

. /appl/profile/zz-csc-env.sh
set -u
module load pytorch/2.5

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
cd "$WORK"

echo "=== E02: Per-language threshold tuning on E01 ==="
echo "Start: $(date)"

python run_threshold_fixed.py \
  --model_dir  artifacts/e01-xlmr-base \
  --dev_tsv    dev.tsv \
  --test_tsv   test.tsv \
  --out_tsv    artifacts/e02-xlmr-threshold_test.tsv \
  --max_length 192 \
  --batch_size 64 \
  --step       0.01 \
  --t_min      0.10 \
  --t_max      0.90 \
  --device     cuda

echo "Done: $(date)"
