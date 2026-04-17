#!/bin/sh
# E03: Twitter-domain XLM-RoBERTa
# Model  : cardiffnlp/twitter-xlm-roberta-base (Twitter-pretrained)
# Data   : train.tsv (EN only, 98k)
# Result : overall F1=0.9066  ENG=0.9433  GER=0.6833  FIN=0.5518
# Note   : Tests whether Twitter-domain pretraining helps with social media text.
#          Weaker on FIN than E01; Twitter vocabulary not multilingual enough.
#SBATCH --job-name=e03-twitter-xlmr
#SBATCH --account=YOUR_CSC_PROJECT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
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

echo "=== E03: twitter-xlm-roberta-base ==="
echo "Start: $(date)"

python run_xlmr.py \
  --model_name  cardiffnlp/twitter-xlm-roberta-base \
  --train_tsv   train.tsv \
  --preset      score \
  --out_dir     artifacts/e03-twitter-xlmr \
  --out_tsv     artifacts/e03-twitter-xlmr_test.tsv \
  --predict_test \
  --hf_home     "$HF_HOME_ROOT"

echo "Done: $(date)"
