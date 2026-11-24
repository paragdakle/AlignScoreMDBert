import argparse
import hashlib
import os
import warnings
from argparse import ArgumentParser

import torch
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from alignscore.dataloader import DSTDataLoader
from alignscore.model import BERTAlignModel

### Original AlignScore training mixture (files produced by generate_training_data.py)
ALL_TRAINING_DATASETS = {
    ### NLI
    'mnli': {'task_type': 'nli', 'data_path': 'mnli.json'},
    'doc_nli': {'task_type': 'bin_nli', 'data_path': 'doc_nli.json'},
    'snli': {'task_type': 'nli', 'data_path': 'snli.json'},
    'anli_r1': {'task_type': 'nli', 'data_path': 'anli_r1.json'},
    'anli_r2': {'task_type': 'nli', 'data_path': 'anli_r2.json'},
    'anli_r3': {'task_type': 'nli', 'data_path': 'anli_r3.json'},

    ### fact checking
    'nli_fever': {'task_type': 'fact_checking', 'data_path': 'nli_fever.json'},
    'vitaminc': {'task_type': 'fact_checking', 'data_path': 'vitaminc.json'},

    ### paraphrase
    'paws': {'task_type': 'paraphrase', 'data_path': 'paws.json'},
    'paws_qqp': {'task_type': 'paraphrase', 'data_path': 'paws_qqp.json'},
    'paws_unlabeled': {'task_type': 'paraphrase', 'data_path': 'paws_unlabeled.json'},
    'qqp': {'task_type': 'paraphrase', 'data_path': 'qqp.json'},
    'wiki103': {'task_type': 'paraphrase', 'data_path': 'wiki103.json'},

    ### QA
    'squad_v2': {'task_type': 'qa', 'data_path': 'squad_v2_new.json'},
    'race': {'task_type': 'qa', 'data_path': 'race.json'},
    'adversarial_qa': {'task_type': 'qa', 'data_path': 'adversarial_qa.json'},
    'drop': {'task_type': 'qa', 'data_path': 'drop.json'},
    'hotpot_qa_distractor': {'task_type': 'qa', 'data_path': 'hotpot_qa_distractor.json'},
    'hotpot_qa_fullwiki': {'task_type': 'qa', 'data_path': 'hotpot_qa_fullwiki.json'},
    'newsqa': {'task_type': 'qa', 'data_path': 'newsqa.json'},
    'quoref': {'task_type': 'qa', 'data_path': 'quoref.json'},
    'ropes': {'task_type': 'qa', 'data_path': 'ropes.json'},
    'boolq': {'task_type': 'qa', 'data_path': 'boolq.json'},
    'eraser_multi_rc': {'task_type': 'qa', 'data_path': 'eraser_multi_rc.json'},
    'quail': {'task_type': 'qa', 'data_path': 'quail.json'},
    'sciq': {'task_type': 'qa', 'data_path': 'sciq.json'},
    'strategy_qa': {'task_type': 'qa', 'data_path': 'strategy_qa.json'},

    ### Coreference
    'gap': {'task_type': 'coreference', 'data_path': 'gap.json'},

    ### Summarization
    'wikihow': {'task_type': 'summarization', 'data_path': 'wikihow.json'},

    ### Information Retrieval
    'msmarco': {'task_type': 'ir', 'data_path': 'msmarco.json'},

    ### STS
    'stsb': {'task_type': 'sts', 'data_path': 'stsb.json'},
    'sick': {'task_type': 'sts', 'data_path': 'sick.json'},
}

### Long-context additions (files produced by prepare_long_context_data.py)
LONG_CONTEXT_TRAINING_DATASETS = {
    'contract_nli': {'task_type': 'nli', 'data_path': 'contract_nli.json'},           # full NDA contracts, ~2.5k tokens
    'control_nli': {'task_type': 'nli', 'data_path': 'control_nli.json'},             # passage-level NLI, ~600 tokens
    'wice': {'task_type': 'bin_nli', 'data_path': 'wice.json'},                       # Wikipedia claims vs cited web pages
    'ragtruth': {'task_type': 'bin_nli', 'data_path': 'ragtruth.json'},               # LLM RAG responses (QA/summ/data2text)
    'minicheck_c2d': {'task_type': 'bin_nli', 'data_path': 'minicheck_c2d.json'},     # MiniCheck synthetic claim->doc
    'minicheck_d2c': {'task_type': 'bin_nli', 'data_path': 'minicheck_d2c.json'},     # MiniCheck synthetic doc->claim
    'factcg_hotpot': {'task_type': 'bin_nli', 'data_path': 'factcg_hotpot.json'},     # FactCG CG2C multi-hop (HotpotQA)
    'factcg_musique': {'task_type': 'bin_nli', 'data_path': 'factcg_musique.json'},   # FactCG CG2C multi-hop (MuSiQue)
    'factcg_doc': {'task_type': 'bin_nli', 'data_path': 'factcg_doc.json'},           # FactCG CG2C document-level
    'longctx_booksum': {'task_type': 'bin_nli', 'data_path': 'longctx_booksum.json'}, # BookSum chapters w/ injected hallucinations
    'quality': {'task_type': 'qa', 'data_path': 'quality.json'},                      # QuALITY long-article multiple choice
    'govreport': {'task_type': 'summarization', 'data_path': 'govreport.json'},       # optional: GovReport (~9k tokens)
}

### Held-out validation sets monitored during training (files in --val-data-path)
HELDOUT_VAL_DATASETS = {
    'contract_nli': {'task_type': 'nli', 'data_path': 'contract_nli.json'},
    'control_nli': {'task_type': 'nli', 'data_path': 'control_nli.json'},
    'wice': {'task_type': 'bin_nli', 'data_path': 'wice.json'},
    'ragtruth': {'task_type': 'bin_nli', 'data_path': 'ragtruth.json'},
    'longctx_booksum': {'task_type': 'bin_nli', 'data_path': 'longctx_booksum.json'},
    'quality': {'task_type': 'qa', 'data_path': 'quality.json'},
    'doc_nli': {'task_type': 'bin_nli', 'data_path': 'doc_nli.json'},
    'llm_aggrefact_dev': {'task_type': 'bin_nli', 'data_path': 'llm_aggrefact_dev.json'},
}

REGISTRY = {**ALL_TRAINING_DATASETS, **LONG_CONTEXT_TRAINING_DATASETS}

PRESETS = {
    'alignscore': list(ALL_TRAINING_DATASETS.keys()),
    'long': [k for k in LONG_CONTEXT_TRAINING_DATASETS if k != 'govreport'],
    'alignscore+long': list(ALL_TRAINING_DATASETS.keys()) + [k for k in LONG_CONTEXT_TRAINING_DATASETS if k != 'govreport'],
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1', 'y'):
        return True
    if v.lower() in ('no', 'false', 'f', '0', 'n'):
        return False
    raise argparse.ArgumentTypeError('boolean value expected')


def model_tag(name):
    """Short name for checkpoints/logs: hub ids keep the org, local paths use the directory name."""
    return os.path.basename(os.path.normpath(name)) if os.path.isdir(name) else name.replace('/', '-')


def interval(v):
    f = float(v)
    return int(f) if f > 1 and f.is_integer() else f


def resolve_files(names, registry, root, size, missing_ok, kind):
    out = {}
    for name in names:
        path = os.path.join(root, registry[name]['data_path'])
        if not os.path.exists(path):
            msg = f"{kind} dataset '{name}' not found at {path}"
            if missing_ok:
                warnings.warn(msg + ' -- skipping')
                continue
            raise FileNotFoundError(msg)
        out[name] = {'task': registry[name]['task_type'], 'size': size, 'data_path': path}
    return out


def train(datasets, heldout, args):
    dm = DSTDataLoader(
        dataset_config=datasets,
        heldout_val_config=heldout,
        model_name=args.model_name,
        sample_mode='seq',
        train_batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        train_eval_split=0.95,
        need_mlm=args.do_mlm,
        tokenizer_max_length=args.tokenizer_max_length,
        max_tokens_per_batch=args.max_tokens_per_batch,
        eval_max_tokens_per_batch=args.eval_max_tokens_per_batch,
        seed=args.seed,
        length_workers=args.length_workers,
        max_train_eval_examples=args.max_train_eval_examples,
    )

    model = BERTAlignModel(model=args.model_name, using_pretrained=args.use_pretrained_model,
        adam_epsilon=args.adam_epsilon,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps_portion=args.warm_up_proportion,
        pooling=args.pooling,
        attn_implementation=args.attn_implementation,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    model.need_mlm = args.do_mlm
    model.val_set_names = list(heldout.keys())
    if not args.do_mlm:
        # MLM head is unused: freeze its own parameters (not the tied embeddings) so DDP sees no unused params
        base_ids = {id(p) for p in model.base_model.parameters()}
        for p in model.mlm_head.parameters():
            if id(p) not in base_ids:
                p.requires_grad_(False)

    datasets_tag = args.dataset_preset or '_'.join(datasets.keys())
    if len(datasets_tag) > 60:
        datasets_tag = 'mix' + hashlib.sha1(datasets_tag.encode()).hexdigest()[:8]
    n_dev = len(args.devices) if isinstance(args.devices, list) else args.devices
    checkpoint_name = '_'.join((
        f"{args.ckpt_comment}{model_tag(args.model_name)}",
        f"{'scratch_' if not args.use_pretrained_model else ''}{'no_mlm_' if not args.do_mlm else ''}{datasets_tag}",
        str(args.max_samples_per_dataset),
        f"{args.max_tokens_per_batch or args.batch_size}{'tok' if args.max_tokens_per_batch else ''}x{n_dev}x{args.accumulate_grad_batch}"
    ))

    monitor, mode = ('val_auc_mean', 'max') if heldout else ('val_loss', 'min')
    best_ckpt = ModelCheckpoint(
        dirpath=args.ckpt_save_path,
        filename=checkpoint_name + "_{epoch:02d}_{step}_best",
        monitor=monitor, mode=mode, save_top_k=2,
        save_on_train_epoch_end=False,
    )
    periodic_ckpt = ModelCheckpoint(
        dirpath=args.ckpt_save_path,
        filename=checkpoint_name + "_{epoch:02d}_{step}",
        every_n_train_steps=args.save_every_n_steps,
        save_top_k=1, save_last=True,
    )

    precision = args.precision or ('bf16-mixed' if model.is_modernbert and args.accelerator == 'gpu' else '32')
    devices = args.devices if args.accelerator == 'gpu' else args.devices[0]  # cpu: number of processes
    n_devices = len(devices) if isinstance(devices, list) else int(devices)
    strategy = DDPStrategy(find_unused_parameters=args.find_unused_parameters) if n_devices > 1 else 'auto'

    trainer = Trainer(
        accelerator=args.accelerator,
        max_epochs=args.num_epoch,
        max_steps=args.max_steps,
        devices=devices,
        strategy=strategy,
        precision=precision,
        callbacks=[best_ckpt, periodic_ckpt, LearningRateMonitor(logging_interval='step')],
        logger=TensorBoardLogger(save_dir=args.log_dir, name=checkpoint_name),
        accumulate_grad_batches=args.accumulate_grad_batch,
        gradient_clip_val=args.gradient_clip_val,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
        log_every_n_steps=args.log_every_n_steps,
        # the token-budget batch sampler shards across ranks itself
        use_distributed_sampler=args.max_tokens_per_batch is None,
    )

    trainer.fit(model, datamodule=dm, ckpt_path=args.resume_from)
    trainer.save_checkpoint(os.path.join(args.ckpt_save_path, f"{checkpoint_name}_final.ckpt"))

    print("Training is finished.")


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='YAML file whose keys (dashes or underscores) set argument defaults')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--batch-size', type=int, default=32, help='sequences per batch (upper bound when --max-tokens-per-batch is set)')
    parser.add_argument('--eval-batch-size', type=int, default=16)
    parser.add_argument('--max-tokens-per-batch', type=int, default=None, help='per-GPU padded-token budget; enables length-bucketed batching')
    parser.add_argument('--eval-max-tokens-per-batch', type=int, default=None)
    parser.add_argument('--tokenizer-max-length', type=int, default=512)
    parser.add_argument('--accumulate-grad-batch', type=int, default=1)
    parser.add_argument('--num-epoch', type=int, default=3)
    parser.add_argument('--max-steps', type=int, default=-1)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--length-workers', type=int, default=8, help='processes used to count tokens (cached once per file)')
    parser.add_argument('--warm-up-proportion', type=float, default=0.06)
    parser.add_argument('--adam-epsilon', type=float, default=1e-6)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--gradient-clip-val', type=float, default=1.0)
    parser.add_argument('--val-check-interval', type=interval, default=1. / 4, help='fraction of an epoch (<=1) or a number of batches (>1)')
    parser.add_argument('--limit-val-batches', type=float, default=1.0)
    parser.add_argument('--max-train-eval-examples', type=int, default=50000, help='cap on the in-pool 5%% validation split')
    parser.add_argument('--log-every-n-steps', type=int, default=50)
    parser.add_argument('--save-every-n-steps', type=int, default=10000)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', nargs='+', type=int, default=None)
    parser.add_argument('--precision', type=str, default=None, help="default: bf16-mixed for ModernBERT, 32 otherwise")
    parser.add_argument('--find-unused-parameters', type=str2bool, default=False)
    parser.add_argument('--model-name', type=str, default="roberta-large")
    parser.add_argument('--pooling', type=str, default='auto', choices=['auto', 'mean', 'cls'], help='ModernBERT only')
    parser.add_argument('--attn-implementation', type=str, default=None, help='ModernBERT only: flash_attention_2 | sdpa | eager')
    parser.add_argument('--gradient-checkpointing', type=str2bool, default=False)
    parser.add_argument('--ckpt-save-path', type=str, default=None)
    parser.add_argument('--ckpt-comment', type=str, default="")
    parser.add_argument('--log-dir', type=str, default='./lightning_logs')
    parser.add_argument('--resume-from', type=str, default=None)
    parser.add_argument('--dataset-preset', type=str, default=None, choices=list(PRESETS.keys()))
    parser.add_argument('--training-datasets', '--trainin-datasets', dest='training_datasets', nargs='+', type=str, default=None, choices=list(REGISTRY.keys()))
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--val-data-path', type=str, default=None, help='directory of held-out validation sets (prepare_long_context_data.py output)')
    parser.add_argument('--val-datasets', nargs='+', type=str, default=None, choices=list(HELDOUT_VAL_DATASETS.keys()))
    parser.add_argument('--max-val-samples-per-dataset', type=int, default=2000)
    parser.add_argument('--max-samples-per-dataset', type=int, default=500000)
    parser.add_argument('--skip-missing-datasets', type=str2bool, default=False)
    parser.add_argument('--do-mlm', type=str2bool, default=False)
    parser.add_argument('--use-pretrained-model', type=str2bool, default=True)
    return parser


def parse_args(argv=None):
    parser = build_parser()
    pre, _ = parser.parse_known_args(argv)
    if pre.config:
        with open(pre.config) as f:
            cfg = yaml.safe_load(f) or {}
        cfg = {k.replace('-', '_'): v for k, v in cfg.items()}
        known = {a.dest for a in parser._actions}
        unknown = set(cfg) - known
        if unknown:
            parser.error(f'unknown keys in {pre.config}: {sorted(unknown)}')
        parser.set_defaults(**cfg)
    args = parser.parse_args(argv)
    for required in ('devices', 'ckpt_save_path', 'data_path'):
        if getattr(args, required) is None:
            parser.error(f"--{required.replace('_', '-')} is required (on the command line or in --config)")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.accelerator == 'gpu' and not torch.cuda.is_available():
        raise SystemExit('No CUDA device found. Use --accelerator cpu --devices 1 for a smoke test.')
    seed_everything(args.seed)
    torch.set_float32_matmul_precision('high')

    names = args.training_datasets or PRESETS[args.dataset_preset or 'alignscore']
    datasets = resolve_files(names, REGISTRY, args.data_path, args.max_samples_per_dataset,
                             args.skip_missing_datasets, 'training')

    heldout = {}
    if args.val_data_path:
        val_names = args.val_datasets or list(HELDOUT_VAL_DATASETS.keys())
        heldout = resolve_files(val_names, HELDOUT_VAL_DATASETS, args.val_data_path, args.max_val_samples_per_dataset,
                                missing_ok=args.val_datasets is None, kind='validation')

    print(f"Training on {len(datasets)} datasets: {', '.join(datasets)}")
    if heldout:
        print(f"Held-out validation sets: {', '.join(heldout)}")
    train(datasets, heldout, args)
