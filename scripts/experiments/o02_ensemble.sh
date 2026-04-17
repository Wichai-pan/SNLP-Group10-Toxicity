#!/bin/sh
# O02: Ensemble L02 + L03 ⭐ CURRENT BEST
# Models : L02 (large-aug) + L03 (large-germeval), averaged softmax probabilities
# Result : overall F1=0.9230  ENG=0.9491  GER=0.7600  FIN=0.8513
# Note   : L02 strong on FIN (0.8477), L03 strong on GER (0.7409).
#          Ensemble combines complementary strengths; all languages improve.
#          No new training needed — inference + probability averaging only.
#          Prerequisites: l02_large_aug.sh + l03_large_germeval.sh completed.
#SBATCH --job-name=o02-ensemble
#SBATCH --account=YOUR_CSC_PROJECT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
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
cd "$WORK"

echo "=== O02: Ensemble L02 (large-aug) + L03 (large-germeval) ==="
echo "Start: $(date)"

python run_ensemble.py \
  --model_dir_a  artifacts/l03-large-germeval \
  --model_dir_b  artifacts/l02-large-aug \
  --dev_tsv      dev.tsv \
  --test_tsv     test.tsv \
  --out_tsv      artifacts/o02-ensemble_test.tsv \
  --max_length   192 \
  --batch_size   32 \
  --t_min        0.10 \
  --t_max        0.90 \
  --step         0.01 \
  --device       cuda

echo "Done: $(date)"
