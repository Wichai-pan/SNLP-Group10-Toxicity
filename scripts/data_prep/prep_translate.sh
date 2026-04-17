#!/bin/sh
# DATA PREP: Translate-Train
# Translates train.tsv EN→DE and EN→FI using Helsinki-NLP opus-mt models.
# Produces: train_de.tsv, train_fi.tsv, train_aug.tsv (~235k rows, 3x original)
# Required by: e05, l02, l05
#SBATCH --job-name=prep-translate
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

cd "$WORK"
mkdir -p logs

. /appl/profile/zz-csc-env.sh
set -u
module load pytorch/2.5

export HF_HOME="$HF_HOME_ROOT"
export TRANSFORMERS_CACHE="$HF_HOME_ROOT/transformers"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME_ROOT"

pip install -q sentencepiece sacremoses 2>/dev/null || true

echo "=== DATA PREP: Translate train.tsv EN→DE/FI ==="
echo "Start: $(date)"

python translate_dataset.py \
    --train_tsv  train.tsv \
    --out_dir    . \
    --batch_size 256 \
    --max_length 192 \
    --device     auto \
    --langs      de,fi \
    --skip_existing

echo "Output: train_de.tsv, train_fi.tsv, train_aug.tsv"
echo "Done: $(date)"
