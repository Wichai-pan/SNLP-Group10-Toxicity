#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv-xlmr}"
HF_HOME_DIR="${HF_HOME_DIR:-artifacts/hf_home}"
DEVICE="${DEVICE:-cpu}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Missing venv python at: $VENV_DIR/bin/python"
  echo "Run: scripts/setup_mac_venv_xlmr.sh"
  exit 1
fi

"$VENV_DIR/bin/python" run_xlmr.py --preset fast --device "$DEVICE" --hf_home "$HF_HOME_DIR" --limit_train 512 --limit_dev 256 --epochs 1
