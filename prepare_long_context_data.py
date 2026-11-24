"""Download and convert long-context factual-consistency datasets into AlignScore's JSONL format.

Writes one JSONL file per dataset into
    <out>/training/<name>.json   training examples (merged with the original AlignScore mixture by train.py)
    <out>/val/<name>.json        held-out validation sets, monitored during training
    <out>/test/<name>.json       test sets, used by evaluate_long_context.py only

Every line is {"task", "text_a", "text_b": [...], "text_c": [...], "orig_label"} as in generate_training_data.py:
    nli      orig_label 0 entailment / 1 neutral / 2 contradiction
    bin_nli  orig_label 1 aligned (supported) / 0 not aligned
    qa       text_b=[question], text_c=[candidate answer], orig_label 1 correct / 0 wrong
    summarization  text_b=[reference summary], negatives sampled at train time

Val/test examples whose context also appears in any training file under <out>/training are dropped
(for our own splits) or flagged with "overlaps_train" (for external benchmarks), see --decontaminate.

Usage:
    python prepare_long_context_data.py --out data            # everything that is public
    python prepare_long_context_data.py --only contract_nli wice
    huggingface-cli login && python prepare_long_context_data.py --only llm_aggrefact   # gated
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import urllib.request
from collections import Counter, defaultdict

FACTCG_REV = '41f185413312ed9b3904ec03963fe899693498e0'
LONGCTX_REV = '30180339ae8ebaa4bdbc361fcc5c648f8144be6d'
FACTCG_URL = f'https://raw.githubusercontent.com/derenlei/FactCG/{FACTCG_REV}/data/'
LONGCTX_URL = f'https://media.githubusercontent.com/media/amazon-science/long-context-hallucination-detection/{LONGCTX_REV}/data/'

TRI = {'entailment': 0, 'neutral': 1, 'contradiction': 2}


def ex(task, text_a, text_b, label, text_c=None, **extra):
    row = {'task': task, 'text_a': text_a, 'text_b': [text_b] if isinstance(text_b, str) else list(text_b),
           'text_c': list(text_c or []), 'orig_label': label}
    row.update(extra)
    return row


def group_split(rows, key, frac, seed=13):
    """Split rows into (big, small) so that no group (e.g. a source document) is in both parts."""
    groups = sorted({key(r) for r in rows})
    rng = random.Random(seed)
    rng.shuffle(groups)
    held = set(groups[:max(1, int(len(groups) * frac))])
    return [r for r in rows if key(r) not in held], [r for r in rows if key(r) in held]


def sample(rows, n, seed=13, stratify=True):
    if n is None or len(rows) <= n:
        return rows
    rng = random.Random(seed)
    if not stratify:
        return rng.sample(rows, n)
    by = defaultdict(list)
    for r in rows:
        by[r['orig_label']].append(r)
    out = []
    for lab, items in sorted(by.items(), key=lambda kv: str(kv[0])):
        k = round(n * len(items) / len(rows))
        out.extend(rng.sample(items, min(k, len(items))))
    rng.shuffle(out)
    return out


def text_hash(s):
    return hashlib.sha1(' '.join(s.lower().split()).encode('utf8')).hexdigest()


def fetch(url, dest):
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f'  downloading {url}')
        tmp = dest + '.part'
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dest)
    return dest


def read_json_or_jsonl(path):
    with open(path, encoding='utf8') as f:
        head = f.read(1)
        f.seek(0)
        if head == '[':
            return json.load(f)
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------------------------------------
# builders: each returns {'training': [...], 'val': [...], 'test': [...]} (any subset of keys)
# ---------------------------------------------------------------------------------------------

def build_contract_nli(a):
    from datasets import load_dataset
    ds = load_dataset('tasksource/contract-nli', 'contractnli_b')  # contractnli_b = full contracts
    names = ds['train'].features['label'].names
    conv = lambda split: [ex('nli', r['premise'], r['hypothesis'], TRI[names[r['label']]]) for r in ds[split]]
    return {'training': conv('train'), 'val': conv('validation'), 'test': conv('test')}


def build_control_nli(a):
    from datasets import load_dataset
    ds = load_dataset('tasksource/ConTRoL-nli')
    conv = lambda split: [ex('nli', r['premise'], r['hypothesis'], TRI[r['label'].strip().lower()]) for r in ds[split]]
    return {'training': conv('train'), 'val': conv('validation'), 'test': conv('test')}


def build_wice(a):
    from datasets import load_dataset
    ds = load_dataset('tasksource/wice')

    def conv(split):
        rows = []
        for r in ds[split]:
            evidence = '\n'.join(s for s in r['evidence'] if s and s.strip())
            # supported -> aligned; partially_supported / not_supported -> not aligned (WiCE's binary setting)
            rows.append(ex('bin_nli', evidence, r['claim'], int(r['label'] == 'supported')))
        return rows
    return {'training': conv('train'), 'val': conv('validation'), 'test': conv('test')}


def build_ragtruth(a):
    from datasets import load_dataset
    ds = load_dataset('wandb/RAGTruth-processed')

    def conv(split):
        rows = []
        for r in ds[split]:
            if r['quality'] != 'good':  # drop truncated outputs and refusals
                continue
            h = r['hallucination_labels_processed']
            label = int((h['evident_conflict'] + h['baseless_info']) == 0)
            context = r['context'] if r['task_type'] == 'Summary' else f"{r['query']}\n\n{r['context']}"
            rows.append(ex('bin_nli', context, r['output'], label, subset=r['task_type']))
        return rows
    train, val = group_split(conv('train'), key=lambda r: text_hash(r['text_a']), frac=0.05)
    return {'training': train, 'val': val, 'test': conv('test')}


def build_minicheck(a):
    from datasets import load_dataset
    out = {}
    for split in ('c2d', 'd2c'):
        ds = load_dataset('lytang/C2D-and-D2C-MiniCheck', split=split)
        out[f'minicheck_{split}'] = {'training': [ex('bin_nli', r['doc'], r['claim'], int(r['label'])) for r in ds]}
    return out


def build_factcg(a):
    files = {
        'factcg_hotpot': 'training_stage1/CG2C_hotpot_qa_rbt_mnli_failed.json',
        'factcg_musique': 'training_stage1/CG2C_musique_minhop3_rbt_mnli_failed.json',
        'factcg_doc': 'training_stage2/CG2C_doc.json',
    }
    out = {}
    for name, rel in files.items():
        path = fetch(FACTCG_URL + rel, os.path.join(a.cache, 'factcg', rel))
        rows = [ex('bin_nli', r['text_a'], r['text_b'], int(r['orig_label']), r.get('text_c'))
                for r in read_json_or_jsonl(path)]
        out[name] = {'training': rows}
    return out


def build_longctx_booksum(a):
    def conv(fname):
        path = fetch(LONGCTX_URL + fname, os.path.join(a.cache, 'longctx', fname))
        # labels: 1 = hallucinated -> orig_label 0 (not aligned)
        return [ex('bin_nli', r['context'], r['response'], 1 - int(r['labels'])) for r in read_json_or_jsonl(path)]
    # the release has no book/chapter ids: group by chapter text so a chapter never spans train and val
    train, val = group_split(conv('train_all.json'), key=lambda r: text_hash(r['text_a']), frac=0.05)
    return {'training': train, 'val': val, 'test': conv('test_all.json')}


def build_quality(a):
    from datasets import load_dataset
    ds = load_dataset('emozilla/quality')

    def conv(split):
        rows = []
        for r in ds[split]:
            for i, opt in enumerate(r['options']):
                rows.append(ex('qa', r['article'], r['question'], int(i == r['answer']), [opt]))
        return rows
    # official test labels are hidden: validation -> test, 10% of training articles -> val
    train, val = group_split(conv('train'), key=lambda r: text_hash(r['text_a']), frac=0.10)
    return {'training': train, 'val': val, 'test': conv('validation')}


def build_govreport(a):
    from datasets import load_dataset
    ds = load_dataset('ccdv/govreport-summarization', 'document', split='train')
    return {'training': [ex('summarization', r['report'], r['summary'], -1) for r in ds]}


def build_doc_nli(a):
    from datasets import load_dataset
    ds = load_dataset('tasksource/doc-nli')
    norm = lambda s: s.strip().strip("'\"").strip()

    def conv(split):
        return [ex('bin_nli', norm(r['premise']), norm(r['hypothesis']), int(norm(r['label']) == 'entailment')) for r in ds[split]]
    out = {'val': sample(conv('validation'), a.max_val), 'test': sample(conv('test'), a.max_test)}
    # only write the training file if the original pipeline did not already produce one
    if a.overwrite or not os.path.exists(os.path.join(a.out, 'training', 'doc_nli.json')):
        out['training'] = conv('train')
    return out


def build_summedits(a):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download('Salesforce/summedits', 'summedits.json', repo_type='dataset')
    rows = [ex('bin_nli', r['doc'], r['summary'], int(r['label']), subset=r['domain']) for r in read_json_or_jsonl(path)]
    return {'test': rows}


def build_llm_aggrefact(a):
    from datasets import load_dataset
    try:
        ds = load_dataset('lytang/LLM-AggreFact')
    except Exception as e:
        raise RuntimeError('lytang/LLM-AggreFact is gated: accept its terms on huggingface.co and run '
                           '`huggingface-cli login` first') from e
    conv = lambda split: [ex('bin_nli', r['doc'], r['claim'], int(r['label']), subset=r['dataset']) for r in ds[split]]
    return {'llm_aggrefact': {'test': conv('test')}, 'llm_aggrefact_dev': {'val': conv('dev')}}


BUILDERS = {
    'contract_nli': build_contract_nli,
    'control_nli': build_control_nli,
    'wice': build_wice,
    'ragtruth': build_ragtruth,
    'minicheck': build_minicheck,
    'factcg': build_factcg,
    'longctx_booksum': build_longctx_booksum,
    'quality': build_quality,
    'doc_nli': build_doc_nli,
    'summedits': build_summedits,
    'llm_aggrefact': build_llm_aggrefact,
    'govreport': build_govreport,
}
DEFAULT = [k for k in BUILDERS if k not in ('llm_aggrefact', 'govreport')]
# external benchmarks: overlapping examples are flagged, not dropped, so scores stay comparable to published ones
EXTERNAL_BENCHMARKS = {'llm_aggrefact', 'llm_aggrefact_dev', 'summedits'}


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def training_context_hashes(train_dir):
    hashes = set()
    for fname in sorted(os.listdir(train_dir)):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(train_dir, fname), encoding='utf8') as f:
            for line in f:
                try:
                    hashes.add(text_hash(json.loads(line)['text_a']))
                except (json.JSONDecodeError, KeyError):
                    break  # not an AlignScore JSONL file
    return hashes


def word_stats(rows):
    lens = sorted(len(r['text_a'].split()) for r in rows)
    if not lens:
        return {}
    q = lambda p: lens[min(len(lens) - 1, int(p * len(lens)))]
    return {'n': len(rows), 'context_words_p50': q(0.5), 'context_words_p90': q(0.9), 'context_words_max': lens[-1],
            'labels': dict(Counter(str(r['orig_label']) for r in rows))}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default='data')
    p.add_argument('--cache', default='data/.raw_cache')
    p.add_argument('--only', nargs='+', choices=list(BUILDERS), default=None)
    p.add_argument('--with-govreport', action='store_true', help='also build the (large) GovReport summarization set')
    p.add_argument('--max-val', type=int, default=5000, help='cap for very large val splits (DocNLI)')
    p.add_argument('--max-test', type=int, default=10000, help='cap for very large test splits (DocNLI)')
    p.add_argument('--decontaminate', choices=['auto', 'report', 'off'], default='auto',
                   help='auto: drop val/test rows whose context is in any training file (flag them for external benchmarks)')
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()

    names = a.only or (DEFAULT + (['govreport'] if a.with_govreport else []))
    built = {}  # dataset name -> {split: rows}
    for name in names:
        print(f'[{name}]')
        try:
            res = BUILDERS[name](a)
        except Exception as e:
            print(f'  FAILED: {e}', file=sys.stderr)
            continue
        if res and all(isinstance(v, dict) for v in res.values()):
            built.update(res)
        else:
            built[name] = res

    for name, splits in built.items():
        # e.g. 5.4k of FactCG's MuSiQue rows have an empty context (almost all labelled "supported")
        for split, rows in list(splits.items()):
            kept = [r for r in rows if r['text_a'].strip() and r['text_b'] and r['text_b'][0].strip()]
            if len(kept) < len(rows):
                print(f'  {name}/{split}: dropped {len(rows) - len(kept)} rows with an empty context or claim')
            splits[split] = kept
        if 'training' in splits:
            write_jsonl(os.path.join(a.out, 'training', f'{name}.json'), splits['training'])

    hashes = set()
    if a.decontaminate != 'off' and os.path.isdir(os.path.join(a.out, 'training')):
        print('hashing training contexts for decontamination...')
        hashes = training_context_hashes(os.path.join(a.out, 'training'))

    manifest_path = os.path.join(a.out, 'long_context_manifest.json')
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    for name, splits in built.items():
        manifest[name] = {}
        for split, rows in splits.items():
            if split != 'training' and hashes:
                overlap = [text_hash(r['text_a']) in hashes for r in rows]
                n_overlap = sum(overlap)
                if n_overlap:
                    if name in EXTERNAL_BENCHMARKS or a.decontaminate == 'report':
                        for r, o in zip(rows, overlap):
                            r['overlaps_train'] = o
                        print(f'  {name}/{split}: {n_overlap} rows share a context with training data (flagged)')
                    else:
                        rows = [r for r, o in zip(rows, overlap) if not o]
                        print(f'  {name}/{split}: dropped {n_overlap} rows whose context appears in training data')
                write_jsonl(os.path.join(a.out, split, f'{name}.json'), rows)
                manifest[name][split] = {**word_stats(rows), 'overlaps_train': n_overlap}
            else:
                if split != 'training':
                    write_jsonl(os.path.join(a.out, split, f'{name}.json'), rows)
                manifest[name][split] = word_stats(rows)
            print(f"  {name}/{split}: {manifest[name][split].get('n', 0)} rows, "
                  f"context p50/p90 words {manifest[name][split].get('context_words_p50')}/{manifest[name][split].get('context_words_p90')}")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'wrote {manifest_path}')


if __name__ == '__main__':
    main()
