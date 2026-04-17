# Puhti experiment plan

## Goal

Move the multilingual toxicity experiments to Puhti and run the final XLM-R training on a single V100 GPU with isolated code, environment, cache, and outputs.

## Remote layout

- Code: `/scratch/YOUR_PROJECT_ID/YOUR_USERNAME/SNLP_Group_Work`
- Python stack: `module load pytorch/2.5`
- Hugging Face cache: `/scratch/YOUR_PROJECT_ID/YOUR_USERNAME/hf-cache/snlp-toxicity`
- Slurm logs: `/scratch/YOUR_PROJECT_ID/YOUR_USERNAME/runs/snlp-toxicity`

## One-time setup

1. Sync the local repository:
   `scripts/sync_to_puhti.sh`
2. SSH to Puhti:
   `ssh puhti`
3. Verify the remote module-based environment:
   `sh /scratch/YOUR_PROJECT_ID/YOUR_USERNAME/SNLP_Group_Work/scripts/setup_puhti_env.sh`

## Experiment order

### E1. Full-train reference run

Use the current main model and full English training data.

- Command:
  `sbatch --job-name=xlmr-score /scratch/YOUR_PROJECT_ID/YOUR_USERNAME/SNLP_Group_Work/scripts/puhti_xlmr_job.sh`
- Expected output:
  `artifacts/xlmr-score/`
  `artifacts/xlmr-score_test.tsv`

### E2. Learning-rate comparison

Run two small variants and compare dev metrics.

- Variant A:
  `sbatch --job-name=xlmr-lr1e5 --export=ALL,RUN_NAME=xlmr-lr1e5,EXTRA_ARGS="--lr 1e-5" /scratch/YOUR_PROJECT_ID/YOUR_USERNAME/SNLP_Group_Work/scripts/puhti_xlmr_job.sh`
- Variant B:
  `sbatch --job-name=xlmr-lr2e5 --export=ALL,RUN_NAME=xlmr-lr2e5,EXTRA_ARGS="--lr 2e-5" /scratch/YOUR_PROJECT_ID/YOUR_USERNAME/SNLP_Group_Work/scripts/puhti_xlmr_job.sh`

### E3. Sequence-length comparison

Check whether longer sequences help multilingual transfer enough to justify the extra cost.

- 128 tokens:
  `sbatch --job-name=xlmr-l128 --export=ALL,RUN_NAME=xlmr-l128,EXTRA_ARGS="--max_length 128" /scratch/YOUR_PROJECT_ID/YOUR_USERNAME/SNLP_Group_Work/scripts/puhti_xlmr_job.sh`
- 192 tokens:
  `sbatch --job-name=xlmr-l192 --export=ALL,RUN_NAME=xlmr-l192,EXTRA_ARGS="--max_length 192" /scratch/YOUR_PROJECT_ID/YOUR_USERNAME/SNLP_Group_Work/scripts/puhti_xlmr_job.sh`

### E4. Epoch comparison

If E1-E3 are stable, compare shorter vs longer fine-tuning.

- 2 epochs:
  `sbatch --job-name=xlmr-e2 --export=ALL,RUN_NAME=xlmr-e2,EXTRA_ARGS="--epochs 2" /scratch/YOUR_PROJECT_ID/YOUR_USERNAME/SNLP_Group_Work/scripts/puhti_xlmr_job.sh`
- 4 epochs:
  `sbatch --job-name=xlmr-e4 --export=ALL,RUN_NAME=xlmr-e4,EXTRA_ARGS="--epochs 4" /scratch/YOUR_PROJECT_ID/YOUR_USERNAME/SNLP_Group_Work/scripts/puhti_xlmr_job.sh`

## Selection rule

Use dev results, not test results, to choose the final model.

Primary reading:
- Overall `f1_macro`
- Per-language `f1_macro`
- Per-language `f1_toxic`

Decision rule:
- Reject runs that only improve English while degrading German and Finnish badly.
- Prefer the run with the best multilingual balance, even if overall accuracy is similar.

## Practical notes

- `/projappl/YOUR_PROJECT_ID/panh` should not be used as the main training location because the filesystem is nearly full.
- Keep large caches and model outputs under `/scratch`.
- The cluster already provides the needed PyTorch/Python stack via `module load pytorch/2.5`, so creating a separate venv is unnecessary here.
- Start with one full run before launching a parameter sweep.
