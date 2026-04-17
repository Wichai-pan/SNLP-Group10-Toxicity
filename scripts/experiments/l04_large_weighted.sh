#!/bin/sh
# L04: XLM-RoBERTa-Large + Class-Weighted Loss
# Model  : xlm-roberta-large (560M params)
# Data   : train.tsv (EN only, 98k)
# Loss   : CrossEntropy with auto-balanced class weights
# Result : overall F1=0.9150  ENG≈0.94  GER≈0.71  FIN≈0.77
# Note   : Class weighting helps slightly but less than data augmentation.
#          Large model capacity not fully utilized without richer training data.
#SBATCH --job-name=l04-large-weighted
#SBATCH --account=YOUR_CSC_PROJECT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
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

echo "=== L04: xlm-roberta-large + class-weighted loss ==="
echo "Start: $(date)"

python run_xlmr.py \
  --model_name            xlm-roberta-large \
  --train_tsv             train.tsv \
  --preset                score \
  --class_weight          auto \
  --gradient_checkpointing \
  --train_batch_size      4 \
  --grad_accum_steps      8 \
  --out_dir               artifacts/l04-large-weighted \
  --out_tsv               artifacts/l04-large-weighted_test.tsv \
  --predict_test \
  --hf_home               "$HF_HOME_ROOT"

echo "Done: $(date)"
