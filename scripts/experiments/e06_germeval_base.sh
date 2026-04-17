#!/bin/sh
# E06: XLM-R Base + GermEval 2018 Native German Data
# Model  : xlm-roberta-base
# Data   : train_germeval.tsv (train.tsv + ~8.5k GermEval rows, ~107k total)
# Result : overall F1=0.9087  ENG=0.9402  GER=0.7183  FIN=0.7333
# Note   : GermEval 2018 is real German Twitter toxic data (public dataset).
#          Most effective strategy for improving GER performance on base model.
#          Prerequisite: prep_germeval.sh must have completed.
#SBATCH --job-name=e06-germeval-base
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

echo "=== E06: XLM-R Base + GermEval 2018 ==="
echo "Train rows: $(wc -l < train_germeval.tsv)"
echo "Start: $(date)"

python run_xlmr.py \
  --model_name  xlm-roberta-base \
  --train_tsv   train_germeval.tsv \
  --preset      score \
  --out_dir     artifacts/e06-germeval-base \
  --out_tsv     artifacts/e06-germeval-base_test.tsv \
  --predict_test \
  --hf_home     "$HF_HOME_ROOT"

echo "Done: $(date)"
