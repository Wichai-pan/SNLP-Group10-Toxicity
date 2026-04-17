#!/bin/bash
#SBATCH --job-name=translate
#SBATCH --account=YOUR_CSC_PROJECT
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/translate_%j.out
#SBATCH --error=logs/translate_%j.err

WORK=$(cd "$(dirname "$0")/../.." && pwd)
HF_HOME_ROOT=${SNLP_HF_HOME:-"$HOME/.cache/huggingface-snlp"}

cd "$WORK"
mkdir -p logs "$HF_HOME_ROOT"

. /appl/profile/zz-csc-env.sh
set -euo pipefail
module purge
module load pytorch/2.5

export HF_HOME="$HF_HOME_ROOT"
export TRANSFORMERS_CACHE="$HF_HOME_ROOT/transformers"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

if python - <<'PY'
try:
    import sacremoses  # noqa: F401
    import sentencepiece  # noqa: F401
except Exception:
    raise SystemExit(1)
raise SystemExit(0)
PY
then
  :
else
  pip install --quiet --no-input sentencepiece sacremoses
fi

echo "=== Translate-Train E3 ==="
echo "Start: $(date)"

python -u translate_dataset.py \
  --train_tsv train.tsv \
  --out_dir . \
  --batch_size 256 \
  --max_length 192 \
  --device auto \
  --langs de,fi \
  --log_every 10 \
  --skip_existing \
  ${EXTRA_ARGS:-}

echo "Done: $(date)"
