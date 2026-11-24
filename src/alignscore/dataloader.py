import hashlib
import json
import logging
import math
import os
import random
from multiprocessing import Pool
from typing import Dict, List, Optional, Sized

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# tasks whose text_b is built as "question + ' ' + option"
QA_LIKE_TASKS = {'qa', 'coreference'}


def _b_side_text(example) -> str:
    """Longest second segment the dataset may build for this example (used for length estimates)."""
    text_b = example.get('text_b') or ['']
    text_c = example.get('text_c') or []
    if example['task'] in QA_LIKE_TASKS:
        return text_b[0] + (' ' + max(text_c, key=len) if text_c else '')
    return max(list(text_b) + list(text_c), key=len)


class DSTDataSet(Dataset):
    def __init__(self, dataset, model_name='bert-base-uncased', need_mlm=True, tokenizer_max_length=512, augment=True) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer_max_length = tokenizer_max_length
        self.config = AutoConfig.from_pretrained(model_name)
        self.dataset_type_dict = dict()

        self.dataset = dataset

        self.need_mlm = need_mlm
        self.augment = augment  # random self-alignment / swapping; off for held-out evaluation sets

        self.dataset_type_dict_init()

    def dataset_type_dict_init(self):
        for i, example in enumerate(self.dataset):
            self.dataset_type_dict.setdefault(example['task'], []).append(i)

    def random_word(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of str, tokenized sentence.
        :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
        :return: (list of str, list of int), masked tokens and related labels for LM prediction
        """
        if not self.need_mlm: # disable masked language modeling
            return tokens, [-100] * len(tokens)

        output_label = []
        special_ids = set(self.tokenizer.all_special_ids)

        for i, token in enumerate(tokens):
            if token in special_ids:
                output_label.append(-100) # PAD / CLS / SEP tokens ignored
                continue
            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.tokenizer.mask_token_id

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(self.tokenizer.vocab_size)

                # -> rest 10% randomly keep current token

                # append current token to output (we will predict these later)
                output_label.append(token)
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-100)

        return tokens, output_label

    def _rand(self):
        """Random draw used by augmentations; returns 0 (never triggers) when augmentation is off."""
        return random.random() if self.augment else 0.0

    # Each process_* returns (text_a, text_b, align_label, tri_label, reg_label).
    # -100 marks a head that is not trained by this example.

    def process_nli(self, index):
        example = self.dataset[index]
        text_a = example['text_a']
        text_b = example['text_b'][0]
        tri_label = example['orig_label'] if example['orig_label'] != -1 else 1

        if self._rand() > 0.95: ### random self alignment
            text_b = example['text_a']
            tri_label = 0
        elif example['orig_label'] == 2 and self._rand() > 0.95:
            text_a = example['text_b'][0]
            text_b = example['text_a']
        return text_a, text_b, -100, tri_label, -100.0

    process_fact_checking = process_nli

    def process_paraphrase(self, index):
        example = self.dataset[index]
        text_a = example['text_a']
        text_b = example['text_b'][0]
        label = example['orig_label']

        if self._rand() > 0.95: ### random self alignment
            text_b = example['text_a']
            label = 1
        elif self._rand() > 0.95:
            text_a = example['text_b'][0]
            text_b = example['text_a']
        return text_a, text_b, label, -100, -100.0

    def process_qa(self, index):
        example = self.dataset[index]
        text_a = example['text_a']
        if len(example['text_c']) > 0:
            text_b = example['text_b'][0] + ' ' + example['text_c'][0]
        else:
            text_b = example['text_b'][0]
        return text_a, text_b, example['orig_label'], -100, -100.0

    process_coreference = process_qa

    def process_bin_nli(self, index):
        example = self.dataset[index]
        return example['text_a'], example['text_b'][0], example['orig_label'], -100, -100.0

    def _pos_or_neg(self, index, task, random_positive):
        example = self.dataset[index]
        if random.random() > 0.5: # this will be a positive pair
            pos_id = random.randint(0, len(example['text_b'])-1) if random_positive else 0
            return example['text_a'], example['text_b'][pos_id], 1
        # this will be a negative pair
        if len(example['text_c']) > 0:
            text_b = example['text_c'][random.randint(0, len(example['text_c'])-1)]
        else:
            text_b = self.dataset[random.choice(self.dataset_type_dict[task])]['text_b'][0]
        return example['text_a'], text_b, 0

    def process_summarization(self, index):
        text_a, text_b, label = self._pos_or_neg(index, 'summarization', random_positive=True)
        return text_a, text_b, label, -100, -100.0

    def process_multiple_choice_qa(self, index):
        text_a, text_b, label = self._pos_or_neg(index, 'multiple_choice_qa', random_positive=False)
        return text_a, text_b, label, -100, -100.0

    def process_extractive_qa(self, index):
        text_a, text_b, label = self._pos_or_neg(index, 'extractive_qa', random_positive=True)
        return text_a, text_b, label, -100, -100.0

    def process_ir(self, index):
        example = self.dataset[index]
        text_b = example['text_b'][random.randint(0, len(example['text_b'])-1)]
        return example['text_a'], text_b, example['orig_label'], -100, -100.0

    def process_sts(self, index):
        example = self.dataset[index]
        return example['text_a'], example['text_b'][0], -100, -100, float(example['orig_label'])

    process_ctc = process_sts
    process_wmt = process_sts

    def encode_pair(self, text_a, text_b, max_length):
        try:
            return self.tokenizer(text_a, text_b, max_length=max_length, truncation='only_first')
        except Exception:
            logging.warning('text_b too long...')
            return self.tokenizer(text_a, text_b, max_length=max_length, truncation=True)

    def __getitem__(self, index):
        example = self.dataset[index]
        process = getattr(self, f"process_{example['task']}", None)
        if process is None:
            raise ValueError(f"Unknown task type '{example['task']}'")
        text_a, text_b, align_label, tri_label, reg_label = process(index)

        # Cap at the pre-computed length so token-budget batches stay within budget even when an
        # augmentation (e.g. self-alignment) makes the pair longer than the estimate.
        max_length = self.tokenizer_max_length
        if '_len' in example:
            max_length = max(min(max_length, int(example['_len'])), 16)
        tokenized_pair = self.encode_pair(text_a, text_b, max_length)
        input_ids, mlm_labels = self.random_word(list(tokenized_pair['input_ids']))

        item = {
            'input_ids': input_ids,
            'attention_mask': tokenized_pair['attention_mask'],
            'align_label': align_label,
            'mlm_label': mlm_labels,
            'tri_label': tri_label,
            'reg_label': reg_label,
        }
        if 'token_type_ids' in tokenized_pair:
            item['token_type_ids'] = tokenized_pair['token_type_ids']
        return item

    def __len__(self):
        return len(self.dataset)


class PadCollator:
    """Pads a list of DSTDataSet items to the longest sequence in the batch (dynamic padding)."""

    def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 8) -> None:
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, items: List[dict]) -> Dict[str, torch.Tensor]:
        max_len = max(len(it['input_ids']) for it in items)
        if self.pad_to_multiple_of:
            max_len = int(math.ceil(max_len / self.pad_to_multiple_of) * self.pad_to_multiple_of)

        def pad(key, value):
            return torch.tensor([it[key] + [value] * (max_len - len(it[key])) for it in items], dtype=torch.long)

        batch = {
            'input_ids': pad('input_ids', self.pad_token_id),
            'attention_mask': pad('attention_mask', 0),
            'mlm_label': pad('mlm_label', -100),
            'align_label': torch.tensor([it['align_label'] for it in items], dtype=torch.long),
            'tri_label': torch.tensor([it['tri_label'] for it in items], dtype=torch.long),
            'reg_label': torch.tensor([it['reg_label'] for it in items], dtype=torch.float),
        }
        if 'token_type_ids' in items[0]:
            batch['token_type_ids'] = pad('token_type_ids', 0)
        return batch


class TokenBudgetBatchSampler(Sampler[List[int]]):
    """Groups examples of similar length into batches of at most `max_tokens` padded tokens.

    Long-context training mixes 30-token NLI pairs with 8k-token documents; a fixed batch size either
    OOMs on the long ones or wastes the GPU on the short ones. Examples are shuffled, sorted by length
    inside large buckets, packed under the token budget, and the batches are shuffled again.
    Distributed-aware: every rank gets the same number of batches.
    """

    def __init__(self, lengths, max_tokens: int, max_batch_size: int = 256, shuffle: bool = True,
                 seed: int = 0, num_replicas: int = 1, rank: int = 0, bucket_batches: int = 100) -> None:
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.num_replicas = num_replicas
        self.rank = rank
        self.bucket_size = max_batch_size * bucket_batches
        self.epoch = 0
        self._batches = self._build()

    def _build(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        n = len(self.lengths)
        order = rng.permutation(n) if self.shuffle else np.arange(n)
        batches = []
        for start in range(0, n, self.bucket_size):
            chunk = order[start:start + self.bucket_size]
            chunk = chunk[np.argsort(-self.lengths[chunk], kind='stable')]
            cur, cur_max = [], 0
            for i in chunk:
                length = int(self.lengths[i])
                new_max = max(cur_max, length)
                if cur and (new_max * (len(cur) + 1) > self.max_tokens or len(cur) >= self.max_batch_size):
                    batches.append(cur)
                    cur, new_max = [], length
                cur.append(int(i))
                cur_max = new_max
            if cur:
                batches.append(cur)
        if self.shuffle:
            rng.shuffle(batches)
        if self.num_replicas > 1:
            total = int(math.ceil(len(batches) / self.num_replicas)) * self.num_replicas
            batches = batches + batches[:total - len(batches)]
            batches = batches[self.rank::self.num_replicas]
        return batches

    def set_epoch(self, epoch: int) -> None:
        if epoch != self.epoch:
            self.epoch = epoch
            self._batches = self._build()

    def __iter__(self):
        yield from self._batches

    def __len__(self):
        return len(self._batches)


class PropSampler(Sampler[int]):
    def __init__(self, data_source: Optional[Sized]) -> None:
        super().__init__()
        self.K = 500000
        print("Initializing Prop Sampler")

        self.data_positions = dict()
        for i, example in tqdm(enumerate(data_source), desc="Initializing Sampler"):
            self.data_positions.setdefault(example['dataset_name'], []).append(i)
        self.all_dataset_names = list(self.data_positions.keys())
        self.dataset_lengths = {each:len(self.data_positions[each]) for each in self.data_positions}

        self.dataset_props = {each: min(self.dataset_lengths[each], self.K) for each in self.dataset_lengths}
        self.dataset_props_sum = sum([self.dataset_props[each] for each in self.dataset_props])

        print("Finish Prop Sampler initialization.")

    def __iter__(self):
        iter_list = []
        for each in self.dataset_props:
            iter_list.extend(np.random.choice(self.data_positions[each], size=self.dataset_props[each], replace=False).tolist())

        random.shuffle(iter_list)

        yield from iter_list

    def __len__(self):
        return self.dataset_props_sum


# ---------------------------------------------------------------------------------------------
# length cache
# ---------------------------------------------------------------------------------------------
_WORKER_TOKENIZER = None


def _init_length_worker(model_name):
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(model_name)


def _length_worker(args):
    texts_a, texts_b = args
    tok = _WORKER_TOKENIZER
    la = tok(texts_a, add_special_tokens=False)['input_ids']
    lb = tok(texts_b, add_special_tokens=False)['input_ids']
    extra = tok.num_special_tokens_to_add(pair=True)
    return [len(a) + len(b) + extra for a, b in zip(la, lb)]


def _length_cache_path(data_path, model_name, n_rows):
    st = os.stat(data_path)
    key = f"{os.path.abspath(data_path)}|{st.st_size}|{int(st.st_mtime)}|{model_name}|{n_rows}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(data_path)), '.length_cache')
    return os.path.join(cache_dir, f"{os.path.basename(data_path)}.{digest}.npy")


def read_jsonl(path, limit=None):
    rows = []
    with open(path, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            rows.append(json.loads(line))
    return rows


def compute_lengths(examples, model_name, num_workers=8, chunk=2000):
    """Exact pair token counts (untruncated) for each example."""
    jobs = []
    for s in range(0, len(examples), chunk):
        part = examples[s:s + chunk]
        jobs.append(([e['text_a'] for e in part], [_b_side_text(e) for e in part]))
    out = []
    if num_workers > 1 and len(jobs) > 1:
        with Pool(num_workers, initializer=_init_length_worker, initargs=(model_name,)) as pool:
            for lens in tqdm(pool.imap(_length_worker, jobs), total=len(jobs), desc='Counting tokens', leave=False):
                out.extend(lens)
    else:
        _init_length_worker(model_name)
        for job in tqdm(jobs, desc='Counting tokens', leave=False):
            out.extend(_length_worker(job))
    return np.asarray(out, dtype=np.int32)


def load_with_lengths(data_path, model_name, limit=None, num_workers=8, compute_if_missing=True):
    """Load a JSONL dataset (first `limit` rows) and attach a cached `_len` field to every example."""
    rows = read_jsonl(data_path, limit)
    cache = _length_cache_path(data_path, model_name, len(rows))
    if os.path.exists(cache):
        lens = np.load(cache)
    elif compute_if_missing:
        lens = compute_lengths(rows, model_name, num_workers=num_workers)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        tmp = cache + f'.tmp{os.getpid()}.npy'
        np.save(tmp, lens)
        os.replace(tmp, cache)
    else:
        lens = None
    if lens is not None:
        for row, n in zip(rows, lens):
            row['_len'] = int(n)
    return rows


def _resolve_limit(path, size):
    if size is None:
        return None
    if isinstance(size, int):
        return size
    n = sum(1 for _ in open(path, 'r', encoding='utf8'))
    return int(size * n)


class DSTDataLoader(LightningDataModule):
    def __init__(self, dataset_config, val_dataset_config=None, sample_mode='seq', model_name='bert-base-uncased',
                 is_finetune=False, need_mlm=True, tokenizer_max_length=512, train_batch_size=32, eval_batch_size=4,
                 num_workers=16, train_eval_split=0.95, max_tokens_per_batch=None, eval_max_tokens_per_batch=None,
                 heldout_val_config=None, seed=2022, length_workers=8, max_train_eval_examples=None, **kwargs):
        super().__init__(**kwargs)
        assert sample_mode in ['seq', 'proportion']
        self.sample_mode = sample_mode
        self.dataset_config = dataset_config
        self.val_dataset_config = val_dataset_config
        self.heldout_val_config = heldout_val_config or {}
        self.num_workers = num_workers
        self.train_eval_split = train_eval_split
        self.tokenizer_max_length = tokenizer_max_length
        self.model_name = model_name
        self.max_tokens_per_batch = max_tokens_per_batch
        self.eval_max_tokens_per_batch = eval_max_tokens_per_batch or max_tokens_per_batch
        self.seed = seed
        self.length_workers = length_workers
        self.max_train_eval_examples = max_train_eval_examples

        self.need_mlm = need_mlm
        self.is_finetune = is_finetune

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.collator = PadCollator(self.tokenizer.pad_token_id)

        self.train_bach_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.dataset = None

    @property
    def use_token_budget(self):
        return self.max_tokens_per_batch is not None and self.sample_mode == 'seq'

    @property
    def heldout_val_names(self):
        return list(self.heldout_val_config.keys())

    def _all_files(self):
        files = [(c['data_path'], c.get('size')) for c in self.dataset_config.values()]
        files += [(c['data_path'], c.get('size')) for c in self.heldout_val_config.values()]
        return files

    def prepare_data(self) -> None:
        """Runs once per node before setup(): counts tokens and writes the length caches."""
        AutoTokenizer.from_pretrained(self.model_name)
        if not self.use_token_budget:
            return
        for path, size in self._all_files():
            load_with_lengths(path, self.model_name, limit=_resolve_limit(path, size), num_workers=self.length_workers)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is not None:
            return

        self.init_training_set()

        def make(rows, augment=True):
            return DSTDataSet(dataset=rows, model_name=self.model_name, need_mlm=self.need_mlm,
                              tokenizer_max_length=self.tokenizer_max_length, augment=augment)

        self.dataset = dict()
        if not self.is_finetune:
            split = int(self.train_eval_split * len(self.raw_dataset))
            eval_rows = self.raw_dataset[split:]
            if self.max_train_eval_examples:
                eval_rows = eval_rows[:self.max_train_eval_examples]
            self.dataset['train'] = make(self.raw_dataset[:split])
            self.dataset['test'] = make(eval_rows)
        else:
            self.dataset['train'] = make(self.raw_dataset[:])
            self.dataset['test'] = make(self.val_raw_dataset[:])

        self.dataset['heldout'] = {}
        for name, cfg in self.heldout_val_config.items():
            rows = self._load(cfg['data_path'], cfg.get('size'))
            for r in rows:
                if 'task' in cfg:
                    r['task'] = cfg['task']
            self.dataset['heldout'][name] = make(rows, augment=False)

    def _load(self, path, size):
        limit = _resolve_limit(path, size)
        if self.use_token_budget:
            return load_with_lengths(path, self.model_name, limit=limit, num_workers=self.length_workers)
        return read_jsonl(path, limit)

    def init_training_set(self):
        self.raw_dataset = []
        if self.sample_mode == 'seq':
            for each_dataset, cfg in self.dataset_config.items():
                try:
                    rows = self._load(cfg['data_path'], cfg['size'])
                except Exception as e:
                    raise RuntimeError(f"failed to load data from {cfg['data_path']} ({each_dataset})") from e
                if 'task' in cfg:  # registry task type wins over whatever is stored in the file
                    for r in rows:
                        r['task'] = cfg['task']
                self.raw_dataset.extend(rows)

            random.shuffle(self.raw_dataset)

        elif self.sample_mode == 'proportion':
            for each_dataset in tqdm(self.dataset_config, desc="Loading data from disk..."):
                for jsonobj in read_jsonl(self.dataset_config[each_dataset]['data_path']):
                    jsonobj['dataset_name'] = each_dataset
                    self.raw_dataset.append(jsonobj)

            random.shuffle(self.raw_dataset)

        if self.is_finetune:
            self.val_raw_dataset = []
            for each_dataset, cfg in self.val_dataset_config.items():
                self.val_raw_dataset.extend(read_jsonl(cfg['data_path'], _resolve_limit(cfg['data_path'], cfg['size'])))
            random.shuffle(self.val_raw_dataset)

    def _world(self):
        trainer = getattr(self, 'trainer', None)
        if trainer is not None:
            return trainer.world_size, trainer.global_rank
        return 1, 0

    def _budget_loader(self, ds, max_tokens, shuffle, max_batch_size):
        world, rank = self._world()
        lengths = [min(int(r.get('_len', self.tokenizer_max_length)), self.tokenizer_max_length) for r in ds.dataset]
        sampler = TokenBudgetBatchSampler(lengths, max_tokens=max_tokens, max_batch_size=max_batch_size,
                                          shuffle=shuffle, seed=self.seed, num_replicas=world, rank=rank)
        return DataLoader(ds, batch_sampler=sampler, collate_fn=self.collator, num_workers=self.num_workers,
                          pin_memory=True, persistent_workers=self.num_workers > 0)

    def train_dataloader(self):
        if self.use_token_budget:
            return self._budget_loader(self.dataset['train'], self.max_tokens_per_batch, True, self.train_bach_size)
        if self.sample_mode == 'seq':
            return DataLoader(self.dataset['train'], batch_size=self.train_bach_size, shuffle=True,
                              num_workers=self.num_workers, collate_fn=self.collator)
        elif self.sample_mode == 'proportion':
            return DataLoader(self.dataset['train'], batch_size=self.train_bach_size, collate_fn=self.collator,
                              sampler=PropSampler(self.raw_dataset[:int(self.train_eval_split*len(self.raw_dataset))]), num_workers=self.num_workers)

    def val_dataloader(self):
        sets = [self.dataset['test']] + [self.dataset['heldout'][n] for n in self.heldout_val_names]
        if self.use_token_budget:
            return [self._budget_loader(ds, self.eval_max_tokens_per_batch, False, self.eval_batch_size) for ds in sets]
        return [DataLoader(ds, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers,
                           collate_fn=self.collator) for ds in sets]
