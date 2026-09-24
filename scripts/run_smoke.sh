#!/usr/bin/env bash
# Run WQT20 smoke training + H1 gate verification
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
python -m wqt20.train --n 3000 --epochs 8 --batch-size 64 --lr 1e-4 \
  --out checkpoints/wqt20-smoke
