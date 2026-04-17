#!/bin/sh
set -e

REMOTE_ROOT="${REMOTE_ROOT:-$HOME/SNLP_Group_Work}"
HF_HOME_ROOT="${HF_HOME_ROOT:-$HOME/.cache/huggingface-snlp}"

SLURM_JOB_ID="${SLURM_JOB_ID-}"
export SLURM_JOB_ID
. /appl/profile/zz-csc-env.sh
set -u
module load pytorch/2.5

mkdir -p "$REMOTE_ROOT" "$HF_HOME_ROOT"

"python" -c "import torch, transformers, sklearn, sentencepiece; print('torch', torch.__version__); print('transformers', transformers.__version__)"

printf 'Environment ready.\n'
printf 'Load with: . /appl/profile/zz-csc-env.sh && module load pytorch/2.5\n'
printf 'HF cache: %s\n' "$HF_HOME_ROOT"
