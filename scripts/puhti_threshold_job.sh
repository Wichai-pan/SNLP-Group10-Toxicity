#!/bin/sh
#SBATCH --account=YOUR_PROJECT_ID
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=runs/%x-%j.out

set -e
REMOTE_ROOT="${REMOTE_ROOT:-/scratch/$SBATCH_ACCOUNT/$USER/SNLP_Group_Work}"
. /appl/profile/zz-csc-env.sh
set -u
module load pytorch/2.5
export TOKENIZERS_PARALLELISM=false
cd "$REMOTE_ROOT"

python run_threshold.py   --model_dir artifacts/xlmr-e1   --dev_tsv   dev.tsv              --test_tsv  test.tsv             --out_tsv   artifacts/xlmr-e2_test.tsv   --max_length 192                 --batch_size 64                  --step 0.01                      --t_min 0.25                     --t_max 0.70                     --device cuda
