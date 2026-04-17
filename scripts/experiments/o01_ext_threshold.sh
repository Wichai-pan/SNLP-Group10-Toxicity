#!/bin/sh
# O01: Extended Threshold Tuning on L03
# Base   : L03 checkpoint (large-germeval)
# Change : Extends search range from [0.25,0.70] → [0.10,0.90]
# Result : overall F1=0.9181  ENG=0.9464  GER=0.7491  FIN=0.8077
# Note   : ENG t=0.25 was hitting lower bound in L03. Extending to 0.10
#          revealed FIN also benefits from lower threshold (+0.008 vs L03).
#          Prerequisite: l03_large_germeval.sh must have completed.
#SBATCH --job-name=o01-ext-threshold
#SBATCH --account=YOUR_CSC_PROJECT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
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

echo "=== O01: Extended threshold [0.10,0.90] on L03 ==="
echo "Start: $(date)"

python run_threshold_fixed.py \
  --model_dir  artifacts/l03-large-germeval \
  --dev_tsv    dev.tsv \
  --test_tsv   test.tsv \
  --out_tsv    artifacts/o01-ext-threshold_test.tsv \
  --max_length 192 \
  --batch_size 64 \
  --step       0.01 \
  --t_min      0.10 \
  --t_max      0.90 \
  --device     cuda

echo "Done: $(date)"
