#!/bin/sh
# L07: XLM-RoBERTa-Large + Translate-Train + Suomi24 (Finnish native data)
# Model  : xlm-roberta-large (560M params)
# Data   : train_aug_suomi24.tsv (~237k rows: EN+DE+FI translated + Suomi24 native)
# Result : TBD
# Note   : Extends L02 with ~2.3k native Finnish toxicity examples (TurkuNLP).
#          First experiment with real Finnish text in training data.
#          Hypothesis: even small native FIN data significantly improves FIN F1.
#          Prerequisite: prep_translate.sh + prep_suomi24.sh must have completed.
#SBATCH --job-name=l07-large-suomi24
#SBATCH --account=YOUR_CSC_PROJECT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=08:00:00
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

echo "=== L07: xlm-roberta-large + translate-train + Suomi24 ==="
echo "Train rows: $(wc -l < train_aug_suomi24.tsv)"
echo "Start: $(date)"

python run_xlmr.py \
  --model_name            xlm-roberta-large \
  --train_tsv             train_aug_suomi24.tsv \
  --preset                score \
  --gradient_checkpointing \
  --train_batch_size      4 \
  --grad_accum_steps      8 \
  --out_dir               artifacts/large-suomi24 \
  --out_tsv               artifacts/large-suomi24_test.tsv \
  --predict_test \
  --hf_home               "$HF_HOME_ROOT"

echo "Done: $(date)"
