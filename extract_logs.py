#!/usr/bin/env python3
"""Extract key result lines from Slurm .out logs, strip tqdm noise."""
import os, re, glob

LOG_DIR = os.environ.get("SNLP_RUNS", "runs/snlp-toxicity")
OUT_DIR = os.environ.get("SNLP_WORK", ".") + "/results/logs"
os.makedirs(OUT_DIR, exist_ok=True)

# Patterns to keep
KEEP = re.compile(
    r"(^===|^==|overall|by_lang|eng \||ger \||fin \||"
    r"New best|Saving best|Wrote test|Done:|Start:|Train rows|"
    r"Ensembling|Best thresh|Grid search|thresholds|"
    r"Epoch \d+ done|All done|^--)"
)
# Patterns to always drop (tqdm bars, warnings)
DROP = re.compile(
    r"(it/s|loss=\d|^\s*\d+/\d+.*\[|FutureWarning|DeprecationWarning|"
    r"warn\(|real_accelerator|ds_accelerator|not initialized|newly init|"
    r"TRAIN this model|autocast|GradScaler|^\s*$)"
)

for fpath in sorted(glob.glob(os.path.join(LOG_DIR, "*.out"))):
    basename = os.path.basename(fpath)
    # Derive clean name: strip job ID suffix  e.g. l03-large-germeval-33915858.out
    clean = re.sub(r'-\d{8,}\.out$', '.log', basename)
    # Keep only the "latest" run per experiment (highest job ID)
    # We'll just extract all and let naming handle it

    with open(fpath, errors='replace') as f:
        lines = f.readlines()

    kept = []
    for line in lines:
        line = line.rstrip()
        if DROP.search(line):
            continue
        if KEEP.search(line) or line.startswith("="):
            kept.append(line)

    if not kept:
        continue

    out_path = os.path.join(OUT_DIR, clean)
    with open(out_path, "w") as f:
        f.write("\n".join(kept) + "\n")
    print(f"{basename:55s} -> {clean}  ({len(kept)} lines)")

print(f"\nDone. Logs written to {OUT_DIR}")
