#!/bin/sh
# Data Prep: TurkuNLP Suomi24 Finnish toxicity data
# Downloads native Finnish toxicity annotations from HuggingFace.
#   Source: TurkuNLP/Suomi24-toxicity-annotated (CC BY-SA 4.0, ~2.3k rows)
# Output:
#   fin_suomi24.tsv          — standalone Finnish native data
#   train_aug_suomi24.tsv    — train_aug.tsv + Suomi24
# Prerequisite: prep_translate.sh must have completed (train_aug.tsv must exist)
#SBATCH --job-name=prep-suomi24
#SBATCH --account=YOUR_CSC_PROJECT
#SBATCH --partition=small
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=runs/%x-%j.out

set -e
WORK=$(cd "$(dirname "$0")/../.." && pwd)
HF_HOME_ROOT=${SNLP_HF_HOME:-"$HOME/.cache/huggingface-snlp"}

. /appl/profile/zz-csc-env.sh
set -u
module load pytorch/2.5

export HF_HOME="$HF_HOME_ROOT"
export TRANSFORMERS_CACHE="$HF_HOME_ROOT/transformers"
export PYTHONUNBUFFERED=1
cd "$WORK"

# Ensure datasets library is available
pip install --quiet --user datasets

echo "=== prep_suomi24: Download TurkuNLP Suomi24 ==="
echo "Start: $(date)"

python prepare_suomi24.py \
  --base_tsv      train_aug.tsv \
  --out_suomi24   fin_suomi24.tsv \
  --out_merged    train_aug_suomi24.tsv

echo "Lines in fin_suomi24.tsv:       $(wc -l < fin_suomi24.tsv)"
echo "Lines in train_aug_suomi24.tsv: $(wc -l < train_aug_suomi24.tsv)"
echo "Done: $(date)"
