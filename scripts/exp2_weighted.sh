#!/bin/sh
# Experiment 2: XLM-R base + class-weighted loss (auto balanced)
# Tests whether addressing the 37% toxic vs 25% GER toxic imbalance helps.
#SBATCH --account=YOUR_PROJECT_ID
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=runs/%x-%j.out

set -e
WORK=$(cd "$(dirname "$0")/../.." && pwd)
HF_HOME_ROOT=${SNLP_HF_HOME:-"$HOME/.cache/huggingface-snlp"}
RUN_NAME="${RUN_NAME:-exp2-weighted}"

. /appl/profile/zz-csc-env.sh
set -u
module load pytorch/2.5

export HF_HOME="$HF_HOME_ROOT"
export TRANSFORMERS_CACHE="$HF_HOME_ROOT/transformers"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME_ROOT" "$WORK/artifacts" "$WORK/runs"

cd "$WORK"

echo "=== Experiment 2: XLM-R + class-weighted loss ==="
echo "Start: $(date)"

python run_xlmr.py \
  --model_name   xlm-roberta-base \
  --train_tsv    train.tsv \
  --preset       score \
  --class_weight auto \
  --out_dir      "artifacts/${RUN_NAME}" \
  --out_tsv      "artifacts/${RUN_NAME}_test.tsv" \
  --predict_test \
  --hf_home      "$HF_HOME_ROOT"

echo "Done: $(date)"
