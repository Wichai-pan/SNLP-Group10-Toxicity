#!/bin/sh
# O05: Ensemble L02 + L06 (GAHD German data)
# Models : large-aug (L02, best FIN) + large-gahd (L06, best GER)
# Method : Average softmax probabilities + per-language threshold
# Result : TBD
# Note   : Tests whether GAHD-enhanced L06 improves GER over L03 in ensemble.
#          L06 GER=0.7436 vs L03 GER=0.7376 — marginal single-model gain.
#          Prerequisite: l02, l06 must have completed.
#SBATCH --job-name=o05-ens-gahd
#SBATCH --account=YOUR_CSC_PROJECT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:30:00
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

echo "=== O05: Ensemble L02 (large-aug) + L06 (large-gahd) ==="
echo "Start: $(date)"

python run_ensemble.py \
  --model_dir_a  artifacts/l02-large-aug \
  --model_dir_b  artifacts/large-gahd \
  --dev_tsv      dev.tsv \
  --test_tsv     test.tsv \
  --out_tsv      artifacts/o05-ensemble-gahd_test.tsv \
  --max_length   192 \
  --batch_size   32 \
  --t_min        0.10 \
  --t_max        0.90 \
  --step         0.01 \
  --device       cuda

echo "Done: $(date)"
