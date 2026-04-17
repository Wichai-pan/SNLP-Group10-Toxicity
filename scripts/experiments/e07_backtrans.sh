#!/bin/sh
# E07: XLM-R Base + Back-Translation Augmentation
# Model  : xlm-roberta-base
# Data   : train_backtrans.tsv (EN + DE→EN + FI→EN paraphrases)
# Result : overall F1=0.9052 (worse than E01!)
# Note   : Back-translation (EN→DE→EN, EN→FI→EN) adds noisy paraphrases.
#          Round-trip translation introduces too much noise; hurts performance.
#          Prerequisite: prep_translate.sh must have completed.
#SBATCH --job-name=e07-backtrans
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

pip install -q sentencepiece sacremoses 2>/dev/null || true

cd "$WORK"

echo "=== E07: XLM-R Base + back-translation ==="
echo "Start: $(date)"

if [ ! -f train_backtrans.tsv ]; then
  echo "--- Running back-translation (DE→EN + FI→EN) ---"
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

python run_xlmr.py \
  --model_name  xlm-roberta-base \
  --train_tsv   train_backtrans.tsv \
  --preset      score \
  --out_dir     artifacts/e07-backtrans \
  --out_tsv     artifacts/e07-backtrans_test.tsv \
  --predict_test \
  --hf_home     "$HF_HOME_ROOT"

echo "Done: $(date)"
