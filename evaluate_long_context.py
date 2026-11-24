"""Evaluate an AlignScore checkpoint on the long-context test sets written by prepare_long_context_data.py.

Reports AUC-ROC and balanced accuracy (threshold 0.5, plus a threshold tuned on the matching val/dev
file when one exists) per test set and per subset (e.g. LLM-AggreFact datasets, SummEdits domains).
For QuALITY it also reports multiple-choice accuracy (argmax over the 4 options).

Compare backbones on the same data:
    python evaluate_long_context.py --model roberta-large --ckpt AlignScore-large.ckpt --modes nli_sp
    python evaluate_long_context.py --model answerdotai/ModernBERT-large --ckpt ckpt/..._best.ckpt --modes nli nli_sp
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

from alignscore.inference import Inferencer


def load(path, limit=None):
    rows = []
    with open(path, encoding='utf8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            rows.append(json.loads(line))
    return rows


def pairs_and_labels(rows):
    premises, hypos, labels = [], [], []
    for r in rows:
        claim = r['text_b'][0]
        if r['task'] in ('qa', 'coreference') and r.get('text_c'):
            claim = claim + ' ' + r['text_c'][0]
        premises.append(r['text_a'])
        hypos.append(claim)
        lab = r['orig_label']
        labels.append(int(lab == 0) if r['task'] in ('nli', 'fact_checking') else int(lab))
    return premises, hypos, np.asarray(labels)


def bacc(labels, scores, thr):
    pred = scores > thr
    pos, neg = labels == 1, labels == 0
    if not pos.any() or not neg.any():
        return float('nan')
    return 0.5 * (pred[pos].mean() + (~pred[neg]).mean())


def best_threshold(labels, scores):
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    return float(max(grid, key=lambda t: bacc(labels, scores, t)))


def metrics(labels, scores, thr_tuned=None):
    out = {'n': int(len(labels)), 'pos_rate': float(labels.mean()) if len(labels) else float('nan')}
    if len(set(labels.tolist())) == 2:
        out['auc'] = float(roc_auc_score(labels, scores))
        out['bacc@0.5'] = float(bacc(labels, scores, 0.5))
        if thr_tuned is not None:
            out['threshold'] = thr_tuned
            out['bacc@tuned'] = float(bacc(labels, scores, thr_tuned))
    return out


def mc_accuracy(rows, scores):
    groups = defaultdict(list)
    for r, s in zip(rows, scores):
        groups[(hash(r['text_a']), r['text_b'][0])].append((s, r['orig_label']))
    hits = [max(g, key=lambda x: x[0])[1] == 1 for g in groups.values() if any(l == 1 for _, l in g)]
    return float(np.mean(hits)) if hits else float('nan')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True)
    p.add_argument('--ckpt', default=None)
    p.add_argument('--test-dir', default='data/test')
    p.add_argument('--val-dir', default='data/val', help='used to tune a per-dataset threshold when a matching file exists')
    p.add_argument('--datasets', nargs='+', default=None, help='test file names without .json (default: all)')
    p.add_argument('--modes', nargs='+', default=['nli_sp'], choices=['nli', 'nli_sp', 'bin', 'bin_sp'])
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-length', type=int, default=None)
    p.add_argument('--chunk-words', type=int, default=None)
    p.add_argument('--limit', type=int, default=None, help='max examples per file (quick runs)')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--exclude-overlapping', action='store_true', help='skip rows flagged overlaps_train')
    p.add_argument('--out', default='exp_results/long_context')
    a = p.parse_args()

    scorer = Inferencer(ckpt_path=a.ckpt, model=a.model, batch_size=a.batch_size, device=a.device,
                        max_length=a.max_length, chunk_words=a.chunk_words, verbose=False)
    print(f'max_length={scorer.max_length} chunk_words={scorer.chunk_words}')

    names = a.datasets or sorted(f[:-5] for f in os.listdir(a.test_dir) if f.endswith('.json'))
    results = {}
    for mode in a.modes:
        scorer.nlg_eval_mode = mode
        for name in names:
            rows = load(os.path.join(a.test_dir, f'{name}.json'), a.limit)
            if a.exclude_overlapping:
                rows = [r for r in rows if not r.get('overlaps_train')]
            prem, hyp, labels = pairs_and_labels(rows)
            scores = scorer.nlg_eval(prem, hyp)[1].numpy()

            # threshold tuned on the dev/val counterpart (llm_aggrefact -> llm_aggrefact_dev)
            thr = None
            for cand in (f'{name}_dev.json', f'{name}.json'):
                vpath = os.path.join(a.val_dir, cand)
                if os.path.exists(vpath):
                    vrows = load(vpath, a.limit)
                    vp, vh, vl = pairs_and_labels(vrows)
                    thr = best_threshold(vl, scorer.nlg_eval(vp, vh)[1].numpy())
                    break

            res = {'all': metrics(labels, scores, thr)}
            subsets = defaultdict(list)
            for i, r in enumerate(rows):
                if 'subset' in r:
                    subsets[r['subset']].append(i)
            for sub, idx in sorted(subsets.items()):
                res[sub] = metrics(labels[idx], scores[idx])
            if rows and rows[0]['task'] == 'qa':
                res['all']['mc_acc'] = mc_accuracy(rows, scores)
            results[f'{mode}/{name}'] = res

            line = res['all']
            print(f"{mode:7s} {name:22s} n={line['n']:6d}  AUC={line.get('auc', float('nan')):.4f}  "
                  f"BAcc@0.5={line.get('bacc@0.5', float('nan')):.4f}"
                  + (f"  BAcc@tuned({thr:.2f})={line['bacc@tuned']:.4f}" if thr is not None and 'bacc@tuned' in line else '')
                  + (f"  MC-acc={line['mc_acc']:.4f}" if 'mc_acc' in line else ''))
            if len(subsets) > 1:
                sub_bacc = [v['bacc@0.5'] for k, v in res.items() if k != 'all' and 'bacc@0.5' in v]
                print(f"{'':7s} {'':22s} mean BAcc@0.5 over {len(sub_bacc)} subsets = {np.mean(sub_bacc):.4f}")

    os.makedirs(a.out, exist_ok=True)
    model_tag = os.path.basename(os.path.normpath(a.model)) if os.path.isdir(a.model) else a.model.replace('/', '-')
    ckpt_tag = os.path.splitext(os.path.basename(a.ckpt))[0] if a.ckpt else 'untrained'
    tag = f'{ckpt_tag[:150]}__{model_tag}'
    out_path = os.path.join(a.out, f'{tag}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
