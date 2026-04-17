#!/bin/sh
# Data Prep: GAHD + GermEval 2021 German data
# Downloads and merges two additional German hate speech datasets:
#   GAHD (~11k, CC-BY-4.0):     https://github.com/jagol/gahd
#   GermEval 2021 (~4.2k):      https://github.com/germeval2021toxic/SharedTask
# Output: train_germeval_gahd.tsv (~122k rows)
# Prerequisite: prep_germeval.sh must have completed (train_germeval.tsv must exist)
#SBATCH --job-name=prep-gahd
#SBATCH --account=YOUR_CSC_PROJECT
#SBATCH --partition=small
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=runs/%x-%j.out

set -e
WORK=$(cd "$(dirname "$0")/../.." && pwd)

. /appl/profile/zz-csc-env.sh
set -u
module load pytorch/2.5

export PYTHONUNBUFFERED=1
cd "$WORK"

echo "=== prep_gahd: Download GAHD + GermEval 2021 ==="
echo "Start: $(date)"

python prepare_gahd.py \
  --base_tsv  train_germeval.tsv \
  --out_tsv   train_germeval_gahd.tsv

echo "Lines in train_germeval_gahd.tsv: $(wc -l < train_germeval_gahd.tsv)"
echo "Done: $(date)"
