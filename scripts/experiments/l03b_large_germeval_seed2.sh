#!/bin/sh
# L03b: XLM-RoBERTa-Large + GermEval 2018 (Seed 2)
# Model  : xlm-roberta-large (560M params)
# Data   : train_germeval.tsv (same as L03)
# Seed   : 123 (L03 uses 42)
# Result : TBD
# Note   : Identical to L03 but different random seed. Used to build
#          multi-seed ensemble (o04) to reduce variance on GER predictions.
#          Prerequisite: prep_germeval.sh must have completed.
#SBATCH --job-name=l03b-germeval-s2
#SBATCH --account=YOUR_CSC_PROJECT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=06:00:00
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

echo "=== L03b: xlm-roberta-large + GermEval 2018 (seed=123) ==="
echo "Train rows: $(wc -l < train_germeval.tsv)"
echo "Start: $(date)"

python run_xlmr.py \
  --model_name            xlm-roberta-large \
  --train_tsv             train_germeval.tsv \
  --preset                score \
  --gradient_checkpointing \
  --train_batch_size      4 \
  --grad_accum_steps      8 \
  --seed                  123 \
  --out_dir               artifacts/l03b-large-germeval \
  --out_tsv               artifacts/l03b-large-germeval_test.tsv \
  --predict_test \
  --hf_home               "$HF_HOME_ROOT"

echo "Done: $(date)"
