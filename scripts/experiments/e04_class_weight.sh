#!/bin/sh
# E04: XLM-R Base + Class-Weighted Loss
# Model  : xlm-roberta-base
# Data   : train.tsv (EN only, 98k)
# Loss   : CrossEntropy with auto-balanced class weights
# Result : overall F1=0.9074  ENG=0.9405  GER=0.6755  FIN=0.7299
# Note   : Addresses 37% toxic train vs 25% GER toxic dev imbalance.
#          Small improvement on FIN; minimal effect on GER.
#SBATCH --job-name=e04-class-weight
#SBATCH --account=YOUR_CSC_PROJECT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
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

echo "=== E04: XLM-R Base + class-weighted loss ==="
echo "Start: $(date)"

python run_xlmr.py \
  --model_name   xlm-roberta-base \
  --train_tsv    train.tsv \
  --preset       score \
  --class_weight auto \
  --out_dir      artifacts/e04-class-weight \
  --out_tsv      artifacts/e04-class-weight_test.tsv \
  --predict_test \
  --hf_home      "$HF_HOME_ROOT"

echo "Done: $(date)"
