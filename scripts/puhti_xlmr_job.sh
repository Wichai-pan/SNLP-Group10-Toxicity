#!/bin/sh
#SBATCH --account=YOUR_CSC_PROJECT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=runs/%x-%j.out

set -e

REMOTE_ROOT="${REMOTE_ROOT:-$HOME/SNLP_Group_Work}"
HF_HOME_ROOT="${HF_HOME_ROOT:-$HOME/.cache/huggingface-snlp}"
RUN_NAME="${RUN_NAME:-xlmr-score}"

. /appl/profile/zz-csc-env.sh
set -u
module load pytorch/2.5

export HF_HOME="$HF_HOME_ROOT"
export TRANSFORMERS_CACHE="$HF_HOME_ROOT/transformers"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$HF_HOME_ROOT" "$REMOTE_ROOT/artifacts" "$HOME/runs/snlp-toxicity"

cd "$REMOTE_ROOT"

python run_xlmr.py \
  --preset "${PRESET:-score}" \
  --device cuda \
  --hf_home "$HF_HOME_ROOT" \
  --train_tsv train.tsv \
  --dev_tsv dev.tsv \
  --test_tsv test.tsv \
  --out_dir "artifacts/${RUN_NAME}" \
  --out_tsv "artifacts/${RUN_NAME}_test.tsv" \
  --predict_test \
  ${EXTRA_ARGS:-}
