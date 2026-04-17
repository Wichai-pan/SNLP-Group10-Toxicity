#!/bin/sh
# E01: XLM-RoBERTa-base Fine-tune
# Model  : xlm-roberta-base (278M params)
# Data   : train.tsv (EN only, 98k)
# Result : overall F1=0.9082  ENG≈0.94  GER≈0.67  FIN≈0.73
# Note   : Zero-shot cross-lingual baseline for XLM-R.
#SBATCH --job-name=e01-xlmr-base
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

echo "=== E01: xlm-roberta-base on train.tsv ==="
echo "Start: $(date)"

python run_xlmr.py \
  --model_name  xlm-roberta-base \
  --train_tsv   train.tsv \
  --preset      score \
  --epochs      5 \
  --out_dir     artifacts/e01-xlmr-base \
  --out_tsv     artifacts/e01-xlmr-base_test.tsv \
  --predict_test \
  --hf_home     "$HF_HOME_ROOT"

echo "Done: $(date)"
