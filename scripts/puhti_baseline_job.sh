#!/bin/sh
#SBATCH --account=YOUR_PROJECT_ID
#SBATCH --partition=small
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=runs/%x-%j.out

set -e

REMOTE_ROOT="${REMOTE_ROOT:-/scratch/$SBATCH_ACCOUNT/$USER/SNLP_Group_Work}"
RUN_NAME="${RUN_NAME:-baseline}"

. /appl/profile/zz-csc-env.sh
set -u
module load pytorch/2.5

mkdir -p "$REMOTE_ROOT/artifacts" "/scratch/$SBATCH_ACCOUNT/$USER/runs/snlp-toxicity"

cd "$REMOTE_ROOT"

python run_baseline.py --train_tsv train.tsv --dev_tsv dev.tsv --test_tsv test.tsv --predict_test --model_out "artifacts/${RUN_NAME}.joblib" --out_tsv "artifacts/${RUN_NAME}_test.tsv"
