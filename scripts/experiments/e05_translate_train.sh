#!/bin/sh
# E05: XLM-R Base + Translate-Train Augmentation
# Model  : xlm-roberta-base
# Data   : train_aug.tsv (EN + translated DE + translated FI, ~235k rows)
# Result : overall F1=0.9088  ENG≈0.94  GER≈0.71  FIN≈0.80
# Note   : Exposes model to DE/FI pseudo-labeled data via machine translation.
#          Helps FIN significantly; GER limited by translation quality.
#          Prerequisite: prep_translate.sh must have completed.
#SBATCH --job-name=e05-translate-train
#SBATCH --account=YOUR_CSC_PROJECT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=runs/%x-%j.out

set -e
WORK=$(cd "$(dirname "$0")/../.." && pwd)
HF_HOME_ROOT=${SNLP_HF_HOME:-"$HOME/.cache/huggingface-snlp"}

. /appl/profile/zz-csc-env.sh
set -u
module load pytorch/2.5

export HF_HOME="$HF_HOME_ROOT"
export TRANSFORMERS_CACHE="$HF_HOME_ROOT/transformers"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME_ROOT" "$WORK/artifacts" "$WORK/runs"
cd "$WORK"

echo "=== E05: XLM-R Base + translate-train ==="
echo "Train rows: $(wc -l < train_aug.tsv)"
echo "Start: $(date)"

python run_xlmr.py \
  --model_name  xlm-roberta-base \
  --train_tsv   train_aug.tsv \
  --preset      score \
  --epochs      5 \
  --out_dir     artifacts/e05-translate-train \
  --out_tsv     artifacts/e05-translate-train_test.tsv \
  --predict_test \
  --hf_home     "$HF_HOME_ROOT"

echo "Done: $(date)"
