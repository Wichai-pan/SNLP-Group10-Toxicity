#!/bin/sh
#SBATCH --job-name=analysis-inference
#SBATCH --account=YOUR_PROJECT_ID
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=runs/%x-%j.out

set -e
WORK=${SNLP_WORK:-"$HOME/SNLP_Group_Work"}
HF_HOME_ROOT=${SNLP_HF_HOME:-"$HOME/.cache/huggingface-snlp"}

. /appl/profile/zz-csc-env.sh
set -u
module load pytorch/2.5

export HF_HOME="$HF_HOME_ROOT"
export TRANSFORMERS_CACHE="$HF_HOME_ROOT/transformers"
export HF_DATASETS_CACHE="$HF_HOME_ROOT/datasets"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
cd "$WORK"

echo "=== Analysis inference: dev + HateCheck DE for L02/L03/L03b ==="
echo "Start: $(date)"

python run_analysis_inference.py \
  --model_dirs  artifacts/l02-large-aug \
                artifacts/l03-large-germeval \
                artifacts/l03b-large-germeval \
  --dev_tsv     dev.tsv \
  --out_dir     artifacts/analysis \
  --batch_size  64 \
  --max_length  192

echo "Done: $(date)"
ls -la artifacts/analysis/
