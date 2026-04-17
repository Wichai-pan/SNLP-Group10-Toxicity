#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY_BIN="${PY_BIN:-/opt/homebrew/bin/python3.11}"
VENV_DIR="${VENV_DIR:-.venv-xlmr}"

if [[ ! -x "$PY_BIN" ]]; then
  echo "Python not found at: $PY_BIN"
  echo "Install with: brew install python@3.11"
  exit 1
fi

echo "Creating venv: $VENV_DIR (python: $PY_BIN)"
"$PY_BIN" -m venv "$VENV_DIR"

echo "Installing dependencies..."
"$VENV_DIR/bin/python" -m pip install -U pip
"$VENV_DIR/bin/pip" install -r requirements.txt

echo "Sanity imports..."
"$VENV_DIR/bin/python" -c "import sklearn; print('sklearn ok')"
"$VENV_DIR/bin/python" -c "import torch; print('torch', torch.__version__); print('mps', getattr(torch.backends,'mps',None) and torch.backends.mps.is_available())"
"$VENV_DIR/bin/python" -c "import transformers; print('transformers', transformers.__version__)"

echo "Done. Activate with: source $VENV_DIR/bin/activate"

