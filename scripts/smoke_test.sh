#!/usr/bin/env bash
# CPU end-to-end check: tiny random ModernBERT, a few real long-context examples, 6 training steps,
# held-out validation, checkpoint reload and scoring. Takes ~1-2 minutes. No GPU or big download needed.
set -euo pipefail
cd "$(dirname "$0")/.."
WORK=${WORK:-/tmp/alignscore_smoke}
rm -rf "$WORK" && mkdir -p "$WORK"
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

python - "$WORK" <<'PY'
import sys, os, json, shutil, glob
from transformers import AutoTokenizer, ModernBertConfig, ModernBertForMaskedLM
work = sys.argv[1]
tok = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-large')
cfg = ModernBertConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=96, num_hidden_layers=3,
                       num_attention_heads=4, max_position_embeddings=8192, pad_token_id=tok.pad_token_id,
                       bos_token_id=tok.cls_token_id, eos_token_id=tok.sep_token_id,
                       cls_token_id=tok.cls_token_id, sep_token_id=tok.sep_token_id)
ModernBertForMaskedLM(cfg).save_pretrained(f'{work}/tiny-modernbert')
tok.save_pretrained(f'{work}/tiny-modernbert')
# a slice of the prepared data, or a synthetic fallback when it has not been prepared yet
for split in ('training', 'val', 'test'):
    os.makedirs(f'{work}/data/{split}', exist_ok=True)
for split, names in {'training': ['contract_nli', 'wice', 'quality', 'minicheck_c2d'], 'val': ['contract_nli', 'wice', 'quality'], 'test': ['contract_nli', 'wice', 'quality']}.items():
    for n in names:
        src = f'data/{split}/{n}.json'
        rows = [json.loads(l) for _, l in zip(range(40), open(src))] if os.path.exists(src) else None
        if not rows:
            rows = [{'task': 'bin_nli', 'text_a': ('The contract lasts five years. ' * (40 * (i % 7 + 1))), 'text_b': ['The contract lasts %s years.' % ('five' if i % 2 else 'two')], 'text_c': [], 'orig_label': i % 2} for i in range(40)]
        with open(f'{work}/data/{split}/{n}.json', 'w') as f:
            for r in rows: f.write(json.dumps(r) + '\n')
PY

python train.py --config configs/smoke_test.yaml --model-name "$WORK/tiny-modernbert" \
  --training-datasets contract_nli wice quality minicheck_c2d \
  --data-path "$WORK/data/training" --val-data-path "$WORK/data/val" \
  --ckpt-save-path "$WORK/ckpt" --log-dir "$WORK/logs"

CKPT=$(ls "$WORK"/ckpt/*_final.ckpt)
python evaluate_long_context.py --model "$WORK/tiny-modernbert" --ckpt "$CKPT" --device cpu \
  --test-dir "$WORK/data/test" --val-dir "$WORK/data/val" --modes nli nli_sp --limit 24 --out "$WORK/results"
echo "SMOKE TEST PASSED"
