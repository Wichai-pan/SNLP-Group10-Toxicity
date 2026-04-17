#!/bin/sh
set -eu

REMOTE_ROOT="${REMOTE_ROOT:-$HOME/SNLP_Group_Work}"

rsync -av \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '.venv-xlmr/' \
  --exclude '__pycache__/' \
  --exclude 'artifacts/' \
  --exclude '*.pyc' \
  ./ puhti:"$REMOTE_ROOT"/

printf 'Synced project to %s\n' "$REMOTE_ROOT"

