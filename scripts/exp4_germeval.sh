#!/bin/sh
# Experiment 4: XLM-R + GermEval 2018 native German data
# Adds ~8500 real German toxic/clean Twitter examples to training.
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
RUN_NAME="${RUN_NAME:-exp4-germeval}"

. /appl/profile/zz-csc-env.sh
set -u
module load pytorch/2.5

export HF_HOME="$HF_HOME_ROOT"
export TRANSFORMERS_CACHE="$HF_HOME_ROOT/transformers"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME_ROOT" "$WORK/artifacts" "$WORK/runs"

cd "$WORK"

echo "=== Experiment 4: GermEval data augmentation ==="
echo "Start: $(date)"

# Step 1: prepare GermEval data (downloads + merges)
if [ ! -f train_germeval.tsv ]; then
  echo "--- Preparing GermEval data ---"
  python prepare_germeval.py \
    --train_tsv  train.tsv \
    --out_dir    . \
    --splits     train,test
fi

echo "train_germeval.tsv rows: $(wc -l < train_germeval.tsv)"

# Step 2: train with merged data
python run_xlmr.py \
  --model_name  xlm-roberta-base \
  --train_tsv   train_germeval.tsv \
  --preset      score \
  --out_dir     "artifacts/${RUN_NAME}" \
  --out_tsv     "artifacts/${RUN_NAME}_test.tsv" \
  --predict_test \
  --hf_home     "$HF_HOME_ROOT"

echo "Done: $(date)"
