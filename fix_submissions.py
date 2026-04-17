#!/usr/bin/env python3
"""
Fix submission TSV files to include all 12,791 test IDs.

Problem: test.tsv has 22 rows with embedded newlines in the text field.
csv.DictReader parses only 12,769 logical rows, but Codabench expects 12,791.

Fix: read all IDs line-by-line with regex, then fill missing IDs with default=0.
"""
import re
import os

WORK = os.environ.get("SNLP_WORK", str(Path(".").resolve()))
TEST_TSV = os.path.join(WORK, "test.tsv")
ARTIFACTS = os.path.join(WORK, "artifacts")

VALID_ID = re.compile(r'^(eng|ger|fin)_test_\d+')

def read_all_test_ids(path):
    """Read all valid test IDs using line-by-line parsing."""
    ids = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n\r').split('\t')
            if parts and VALID_ID.match(parts[0]):
                ids.append(parts[0])
    print(f"Found {len(ids)} valid IDs in test.tsv")
    return ids

def load_predictions(path):
    """Load id->prediction map from a TSV file."""
    preds = {}
    with open(path, 'r', encoding='utf-8') as f:
        header = f.readline().rstrip('\n\r').split('\t')
        # find id and predicted columns
        try:
            id_col = header.index('id')
            pred_col = header.index('predicted')
        except ValueError:
            # fallback: assume 0=id, 1=predicted
            id_col, pred_col = 0, 1
        for line in f:
            parts = line.rstrip('\n\r').split('\t')
            if len(parts) > max(id_col, pred_col):
                preds[parts[id_col]] = parts[pred_col]
    return preds

def fix_submission(src_path, dst_path, all_ids, default='0'):
    """Create fixed submission with all 12,791 IDs."""
    preds = load_predictions(src_path)
    missing = [i for i in all_ids if i not in preds]
    print(f"  {os.path.basename(src_path)}: {len(preds)} predictions, {len(missing)} missing -> filling with {default!r}")
    if missing:
        print(f"  Missing IDs: {missing[:5]}{'...' if len(missing)>5 else ''}")

    with open(dst_path, 'w', encoding='utf-8') as f:
        f.write("id\tpredicted\n")
        for id_ in all_ids:
            pred = preds.get(id_, default)
            f.write(f"{id_}\t{pred}\n")

    # verify
    with open(dst_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    assert len(lines) == len(all_ids) + 1, f"Expected {len(all_ids)+1} lines, got {len(lines)}"
    print(f"  -> Written {len(all_ids)} rows to {os.path.basename(dst_path)}")

def main():
    all_ids = read_all_test_ids(TEST_TSV)
    assert len(all_ids) == 12791, f"Expected 12791 IDs, got {len(all_ids)}"

    # Fixed output directory
    fixed_dir = os.path.join(WORK, "submissions_fixed")
    os.makedirs(fixed_dir, exist_ok=True)

    # All artifact TSVs to fix
    targets = [
        "baseline_test.tsv",
        "exp1-twitter_test.tsv",
        "exp2-weighted_test.tsv",
        "exp3-large_test.tsv",
        "exp4-germeval_test.tsv",
        "exp5-backtrans_test.tsv",
        "large-aug_test.tsv",
        "large-aug-th_test.tsv",
        "large-germeval_test.tsv",
        "large-germeval-th_test.tsv",
        "large-weighted_test.tsv",
        "xlmr-e1_test.tsv",
        "xlmr-e2_test.tsv",
        "xlmr-e3b_test.tsv",
        "xlmr-e3_test.tsv",
        "xlmr-e4_test.tsv",
        "xlmr-score_test.tsv",
    ]

    for name in targets:
        src = os.path.join(ARTIFACTS, name)
        dst = os.path.join(fixed_dir, name)
        if os.path.exists(src):
            fix_submission(src, dst, all_ids)
        else:
            print(f"  SKIP (not found): {name}")

    print(f"\nAll fixed files written to: {fixed_dir}")
    print("Row counts:")
    import subprocess
    result = subprocess.run(['wc', '-l'] + [os.path.join(fixed_dir, t) for t in targets if os.path.exists(os.path.join(fixed_dir, t))],
                           capture_output=True, text=True)
    print(result.stdout)

if __name__ == "__main__":
    main()
