#!/bin/sh
# L06: XLM-RoBERTa-Large + GermEval 2018 + GAHD + GermEval 2021
# Model  : xlm-roberta-large (560M params)
# Data   : train_germeval_gahd.tsv (~122k rows: EN + GermEval2018 + GAHD + GermEval2021)
# Result : TBD
# Note   : Extends L03 with ~15k more native German hate speech examples.
#          GAHD (CC-BY-4.0, ~11k adversarial) + GermEval2021 (~4.2k Facebook).
#          Hypothesis: more diverse German data → better GER generalisation.
#          Prerequisite: prep_germeval.sh + prep_gahd.sh must have completed.
#SBATCH --job-name=l06-large-gahd
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

echo "=== L06: xlm-roberta-large + GermEval2018 + GAHD + GermEval2021 ==="
echo "Train rows: $(wc -l < train_germeval_gahd.tsv)"
echo "Start: $(date)"

python run_xlmr.py \
  --model_name            xlm-roberta-large \
  --train_tsv             train_germeval_gahd.tsv \
  --preset                score \
  --gradient_checkpointing \
  --train_batch_size      4 \
  --grad_accum_steps      8 \
  --out_dir               artifacts/large-gahd \
  --out_tsv               artifacts/large-gahd_test.tsv \
  --predict_test \
  --hf_home               "$HF_HOME_ROOT"

echo "Done: $(date)"
