#!/bin/sh
# O04: Multi-seed Ensemble (L02 + L03 seed42 + L03b seed123)
# Models : large-aug (L02) + large-germeval seed42 (L03) + large-germeval seed123 (L03b)
# Method : Average softmax probabilities from 3 models + per-language threshold
# Result : TBD
# Note   : Two L03 seeds reduce variance on GER predictions. L02 maintains FIN strength.
#          Prerequisite: l02, l03, l03b must have completed.
#SBATCH --job-name=o04-ens-seeds
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

echo "=== O04: Ensemble L02 + L03(seed42) + L03b(seed123) ==="
echo "Start: $(date)"

python run_ensemble_n.py \
  --model_dirs  artifacts/l02-large-aug \
                artifacts/l03-large-germeval \
                artifacts/l03b-large-germeval \
  --dev_tsv     dev.tsv \
  --test_tsv    test.tsv \
  --out_tsv     artifacts/o04-ensemble-seeds-th_test.tsv \
  --batch_size  64 \
  --max_length  192 \
  --t_min       0.10 \
  --t_max       0.90

echo "Done: $(date)"
