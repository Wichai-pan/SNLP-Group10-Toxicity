#!/bin/sh
# Experiment 3: XLM-RoBERTa-large (560M params vs 278M base)
# Tests whether model capacity helps cross-lingual transfer.
#SBATCH --account=YOUR_PROJECT_ID
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=runs/%x-%j.out

set -e
WORK=$(cd "$(dirname "$0")/../.." && pwd)
HF_HOME_ROOT=${SNLP_HF_HOME:-"$HOME/.cache/huggingface-snlp"}
RUN_NAME="${RUN_NAME:-exp3-large}"

. /appl/profile/zz-csc-env.sh
set -u
module load pytorch/2.5

export HF_HOME="$HF_HOME_ROOT"
export TRANSFORMERS_CACHE="$HF_HOME_ROOT/transformers"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME_ROOT" "$WORK/artifacts" "$WORK/runs"

cd "$WORK"

echo "=== Experiment 3: xlm-roberta-large ==="
echo "Start: $(date)"

# large needs gradient checkpointing to fit in 32G V100
python run_xlmr.py \
  --model_name            xlm-roberta-large \
  --train_tsv             train.tsv \
  --preset                score \
  --gradient_checkpointing \
  --train_batch_size      4 \
  --grad_accum_steps      8 \
  --out_dir               "artifacts/${RUN_NAME}" \
  --out_tsv               "artifacts/${RUN_NAME}_test.tsv" \
  --predict_test \
  --hf_home               "$HF_HOME_ROOT"

echo "Done: $(date)"
