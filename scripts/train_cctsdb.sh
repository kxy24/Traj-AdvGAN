#!/usr/bin/env bash
set -e
if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/train_cctsdb.sh /path/to/victim_weights.pt [extra train.py args]"
  exit 1
fi
WEIGHTS="$1"; shift
python train.py --data-config configs/datasets/cctsdb.yaml --attack-config configs/attack.yaml --victim-weights "$WEIGHTS" "$@"
