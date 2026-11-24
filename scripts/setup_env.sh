#!/usr/bin/env bash
# Create the training environment on a CUDA machine.
#   bash scripts/setup_env.sh            # uses python3 on PATH
#   PYTHON=python3.11 bash scripts/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}

$PYTHON -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
# Install the torch build matching your CUDA first if the default wheel does not fit, e.g.
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e .
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# FlashAttention 2: unpadded attention for ModernBERT (strongly recommended for 8k training).
if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  pip install flash-attn --no-build-isolation || echo "flash-attn install failed; training will fall back to sdpa (slower, more memory)"
fi
python -c "import torch, transformers, pytorch_lightning as pl; print('torch', torch.__version__, '| transformers', transformers.__version__, '| lightning', pl.__version__, '| cuda', torch.cuda.is_available())"
