#!/bin/sh
# DATA PREP: GermEval 2018
# Downloads GermEval 2018 German Twitter dataset from GitHub.
# Maps OFFENSE→1 / OTHER→0, merges with train.tsv.
# Produces: train_germeval.tsv (~107k rows)
# Required by: e06, l03, l05
#SBATCH --job-name=prep-germeval
#SBATCH --account=YOUR_CSC_PROJECT
#SBATCH --partition=small
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=runs/%x-%j.out

set -e
WORK=$(cd "$(dirname "$0")/../.." && pwd)

. /appl/profile/zz-csc-env.sh
set -u
module load pytorch/2.5

export PYTHONUNBUFFERED=1
cd "$WORK"

echo "=== DATA PREP: GermEval 2018 ==="
echo "Start: $(date)"

python prepare_germeval.py \
    --train_tsv  train.tsv \
    --out_dir    . \
    --splits     train,test

echo "Output: train_germeval.tsv (~107k rows)"
echo "Done: $(date)"
