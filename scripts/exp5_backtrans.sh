#!/bin/sh
# Experiment 5: Back-translation augmentation (DE->EN + FI->EN)
# Paraphrases of original English training data via forward+back translation.
#SBATCH --account=YOUR_PROJECT_ID
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=runs/%x-%j.out

set -e
WORK=$(cd "$(dirname "$0")/../.." && pwd)
HF_HOME_ROOT=${SNLP_HF_HOME:-"$HOME/.cache/huggingface-snlp"}
RUN_NAME="${RUN_NAME:-exp5-backtrans}"

. /appl/profile/zz-csc-env.sh
set -u
module load pytorch/2.5

export HF_HOME="$HF_HOME_ROOT"
export TRANSFORMERS_CACHE="$HF_HOME_ROOT/transformers"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME_ROOT" "$WORK/artifacts" "$WORK/runs"

pip install -q sentencepiece sacremoses 2>/dev/null || true

cd "$WORK"

echo "=== Experiment 5: Back-translation augmentation ==="
echo "Start: $(date)"

# Step 1: back-translate DE->EN and FI->EN
if [ ! -f train_backtrans.tsv ]; then
  echo "--- Running back-translation ---"
  python back_translate.py \
    --train_tsv     train.tsv \
    --train_de_tsv  train_de.tsv \
    --train_fi_tsv  train_fi.tsv \
    --out_dir       . \
    --batch_size    256 \
    --max_length    192 \
    --device        auto \
    --langs         de,fi
fi

echo "train_backtrans.tsv rows: $(wc -l < train_backtrans.tsv)"

# Step 2: train with back-translated data
python run_xlmr.py \
  --model_name  xlm-roberta-base \
  --train_tsv   train_backtrans.tsv \
  --preset      score \
  --out_dir     "artifacts/${RUN_NAME}" \
  --out_tsv     "artifacts/${RUN_NAME}_test.tsv" \
  --predict_test \
  --hf_home     "$HF_HOME_ROOT"

echo "Done: $(date)"
