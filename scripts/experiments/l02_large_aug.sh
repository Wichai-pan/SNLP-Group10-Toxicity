#!/bin/sh
# L02: XLM-RoBERTa-Large + Translate-Train (L1 submission)
# Model  : xlm-roberta-large (560M params)
# Data   : train_aug.tsv (EN + DE + FI translated, ~235k rows)
# Result : overall F1=0.9148 (+th→0.9155)  ENG=0.9455  GER=0.7080  FIN=0.8477
# Note   : Best FIN performance (0.8477). GER weaker than L03 due to
#          translated DE being lower quality than native GermEval data.
#          Prerequisite: prep_translate.sh must have completed.
#SBATCH --job-name=l02-large-aug
#SBATCH --account=YOUR_CSC_PROJECT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=07:00:00
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

echo "=== L02: xlm-roberta-large + train_aug.tsv ==="
echo "Train rows: $(wc -l < train_aug.tsv)"
echo "Start: $(date)"

python run_xlmr.py \
  --model_name            xlm-roberta-large \
  --train_tsv             train_aug.tsv \
  --preset                score \
  --epochs                3 \
  --gradient_checkpointing \
  --train_batch_size      4 \
  --grad_accum_steps      8 \
  --out_dir               artifacts/l02-large-aug \
  --out_tsv               artifacts/l02-large-aug_test.tsv \
  --predict_test \
  --hf_home               "$HF_HOME_ROOT"

echo "Done: $(date)"
