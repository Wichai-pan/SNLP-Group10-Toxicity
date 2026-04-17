#!/bin/sh
# L05: XLM-RoBERTa-Large + Translate-Train + GermEval (Combined)
# Model  : xlm-roberta-large (560M params)
# Data   : train_aug_germeval.tsv (EN+DE+FI translated + native GermEval, ~244k)
# Result : overall F1=0.9151 (+th→0.9225)  ENG=0.9486  GER=0.7519  FIN=0.8097
#          Threshold: ENG t=0.82, GER t=0.77, FIN t=0.50
# Note   : Combines best data strategies: translate-train (→FIN) + GermEval (→GER).
#          Slightly below O02 ensemble overall; FIN weaker than L02 alone.
#          Prerequisites: prep_translate.sh + prep_germeval.sh + prep_combined.sh
#SBATCH --job-name=l05-large-combined
#SBATCH --account=YOUR_CSC_PROJECT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=12:00:00
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
cd "$WORK"

if [ ! -f train_aug_germeval.tsv ]; then
    echo "Building train_aug_germeval.tsv ..."
    python make_aug_germeval.py
fi

echo "=== L05: xlm-roberta-large + translate-train + GermEval ==="
echo "Train rows: $(wc -l < train_aug_germeval.tsv)"
echo "Start: $(date)"

python run_xlmr.py \
  --model_name            xlm-roberta-large \
  --train_tsv             train_aug_germeval.tsv \
  --preset                score \
  --epochs                4 \
  --gradient_checkpointing \
  --train_batch_size      4 \
  --grad_accum_steps      8 \
  --out_dir               artifacts/l05-large-combined \
  --out_tsv               artifacts/l05-large-combined_test.tsv \
  --predict_test \
  --hf_home               "$HF_HOME_ROOT"

echo "--- Running threshold tuning ---"

python run_threshold_fixed.py \
  --model_dir  artifacts/l05-large-combined \
  --dev_tsv    dev.tsv \
  --test_tsv   test.tsv \
  --out_tsv    artifacts/l05-large-combined-th_test.tsv \
  --max_length 192 \
  --batch_size 64 \
  --step       0.01 \
  --t_min      0.10 \
  --t_max      0.90 \
  --device     cuda

echo "All done: $(date)"
