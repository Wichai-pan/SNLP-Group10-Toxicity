#!/bin/sh
# L01: XLM-RoBERTa-Large (No Augmentation)
# Model  : xlm-roberta-large (560M params, 2x base)
# Data   : train.tsv (EN only, 98k)
# Result : overall F1=0.9165  ENG=0.9440  GER=0.7200  FIN=0.7691
# Note   : Scaling up model capacity alone gives +0.008 over E01 base.
#          Requires gradient_checkpointing to fit in 40GB V100.
#SBATCH --job-name=l01-large-base
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

echo "=== L01: xlm-roberta-large on train.tsv ==="
echo "Start: $(date)"

python run_xlmr.py \
  --model_name            xlm-roberta-large \
  --train_tsv             train.tsv \
  --preset                score \
  --gradient_checkpointing \
  --train_batch_size      4 \
  --grad_accum_steps      8 \
  --out_dir               artifacts/l01-large-base \
  --out_tsv               artifacts/l01-large-base_test.tsv \
  --predict_test \
  --hf_home               "$HF_HOME_ROOT"

echo "Done: $(date)"
