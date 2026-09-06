#!/usr/bin/env bash
set -e
if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/ablation_tt100k_yolov10.sh /path/to/yolov10_weights.pt [extra train.py args]"
  exit 1
fi
WEIGHTS="$1"; shift
COMMON=(--data-config configs/datasets/tt100k.yaml --attack-config configs/attack.yaml --victim-weights "$WEIGHTS")
python train.py "${COMMON[@]}" --disable-trajectory --disable-attention --output outputs/ablation_base "$@"
python train.py "${COMMON[@]}" --disable-attention --output outputs/ablation_trajectory "$@"
python train.py "${COMMON[@]}" --output outputs/ablation_full "$@"
