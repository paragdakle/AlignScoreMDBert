#!/usr/bin/env bash
# Train AlignScore on ModernBERT-large with 8k context.
#   bash scripts/train_modernbert_large.sh                 # all visible GPUs
#   DEVICES="0 1 2 3" bash scripts/train_modernbert_large.sh --learning-rate 4e-5
# Extra arguments override configs/modernbert_large_8k.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
DEVICES=${DEVICES:-$(seq -s ' ' 0 $((NGPU - 1)))}
export TOKENIZERS_PARALLELISM=false
python train.py --config configs/modernbert_large_8k.yaml --devices $DEVICES "$@"
