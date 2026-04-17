#!/bin/sh
# O03: Three-model Ensemble (L02 + L03 + L05)
# Models : large-aug (L02) + large-germeval (L03) + large-combined (L05)
# Method : Average softmax probabilities from 3 models + per-language threshold
# Result : TBD
# Note   : No new training. L05 adds combined-data signal to the L02+L03 ensemble.
#          Prerequisite: l02, l03, l05 must have completed.
#SBATCH --job-name=o03-ensemble3
#SBATCH --account=YOUR_CSC_PROJECT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
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

echo "=== O03: Ensemble L02 + L03 + L05 ==="
echo "Start: $(date)"

python run_ensemble_n.py \
  --model_dirs  artifacts/l02-large-aug \
                artifacts/l03-large-germeval \
                artifacts/l05-large-combined \
  --dev_tsv     dev.tsv \
  --test_tsv    test.tsv \
  --out_tsv     artifacts/o03-ensemble3-th_test.tsv \
  --batch_size  64 \
  --max_length  192 \
  --t_min       0.10 \
  --t_max       0.90

echo "Done: $(date)"
