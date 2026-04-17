#!/bin/sh
# DATA PREP: Combined augmented dataset
# Combines train_aug.tsv (EN+DE+FI translated) with GermEval-only rows.
# Produces: train_aug_germeval.tsv (~244k rows)
# Required by: l05
# Prerequisites: prep_translate.sh, prep_germeval.sh must have run first.
#SBATCH --job-name=prep-combined
#SBATCH --account=YOUR_CSC_PROJECT
#SBATCH --partition=small
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=runs/%x-%j.out

set -e
WORK=$(cd "$(dirname "$0")/../.." && pwd)

. /appl/profile/zz-csc-env.sh
set -u
module load pytorch/2.5

export PYTHONUNBUFFERED=1
cd "$WORK"

echo "=== DATA PREP: Combined aug+germeval dataset ==="
echo "Start: $(date)"

python make_aug_germeval.py

echo "Output: train_aug_germeval.tsv (~244k rows)"
echo "Done: $(date)"
