#!/bin/sh
# L03: XLM-RoBERTa-Large + GermEval 2018 ⭐ BEST SINGLE MODEL
# Model  : xlm-roberta-large (560M params)
# Data   : train_germeval.tsv (train.tsv + ~8.5k native GermEval, ~107k total)
# Result : overall F1=0.9161 (+th→0.9177)  ENG=0.9453  GER=0.7409  FIN=0.7926
#          Threshold: ENG t=0.25, FIN t=0.26, GER t=0.68
# Note   : Best single-model GER performance. Native German Twitter data
#          more effective than machine-translated DE data.
#          Prerequisite: prep_germeval.sh must have completed.
#SBATCH --job-name=l03-large-germeval
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

echo "=== L03: xlm-roberta-large + GermEval 2018 ==="
echo "Train rows: $(wc -l < train_germeval.tsv)"
echo "Start: $(date)"

python run_xlmr.py \
  --model_name            xlm-roberta-large \
  --train_tsv             train_germeval.tsv \
  --preset                score \
  --gradient_checkpointing \
  --train_batch_size      4 \
  --grad_accum_steps      8 \
  --out_dir               artifacts/l03-large-germeval \
  --out_tsv               artifacts/l03-large-germeval_test.tsv \
  --predict_test \
  --hf_home               "$HF_HOME_ROOT"

echo "Done: $(date)"
