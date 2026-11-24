import math
from dataclasses import dataclass
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from transformers import (
    AlbertForMaskedLM,
    AlbertModel,
    AutoConfig,
    AutoModelForMaskedLM,
    BertForPreTraining,
    BertModel,
    RobertaForMaskedLM,
    RobertaModel,
    get_linear_schedule_with_warmup,
)


def is_modernbert(model_name_or_path: str) -> bool:
    """True if the checkpoint is a ModernBERT model (works for hub ids and local dirs)."""
    if 'modernbert' in model_name_or_path.lower():
        return True
    try:
        return AutoConfig.from_pretrained(model_name_or_path).model_type == 'modernbert'
    except Exception:
        return False


def default_attn_implementation() -> str:
    """flash_attention_2 when available (unpadded, fastest for long inputs), otherwise sdpa."""
    try:
        import flash_attn  # noqa: F401
        if torch.cuda.is_available():
            return 'flash_attention_2'
    except ImportError:
        pass
    return 'sdpa'


class SequencePooler(nn.Module):
    """Pooler for encoders without a pretrained pooler (ModernBERT).

    Mirrors transformers' ModernBertPredictionHead (dense -> act -> norm) applied to either
    the [CLS] state or the attention-masked mean of all token states.
    """

    def __init__(self, hidden_size: int, pooling: str = 'mean', norm_eps: float = 1e-5) -> None:
        super().__init__()
        assert pooling in ('mean', 'cls'), f"unknown pooling '{pooling}'"
        self.pooling = pooling
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden_size, eps=norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == 'cls':
            pooled = hidden_states[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.norm(self.act(self.dense(pooled)))


class BERTAlignModel(pl.LightningModule):
    def __init__(self, model='bert-base-uncased', using_pretrained=True, pooling='auto',
                 attn_implementation=None, gradient_checkpointing=False, *args, **kwargs) -> None:
        super().__init__()
        # Already defined in lightning: self.device
        self.save_hyperparameters()
        self.model = model
        self.is_modernbert = is_modernbert(model)
        self.pooler = None  # only set for encoders without a pretrained pooler

        if self.is_modernbert:
            config = AutoConfig.from_pretrained(model)
            attn_impl = attn_implementation or default_attn_implementation()
            if using_pretrained:
                mlm_model = AutoModelForMaskedLM.from_pretrained(model, attn_implementation=attn_impl)
            else:
                mlm_model = AutoModelForMaskedLM.from_config(config, attn_implementation=attn_impl)
            self.base_model = mlm_model.model
            # head (dense/act/norm) + decoder, decoder weight is tied to the token embeddings
            self.mlm_head = nn.Sequential(mlm_model.head, mlm_model.decoder)
            if pooling == 'auto':
                pooling = getattr(config, 'classifier_pooling', 'mean')
            self.pooler = SequencePooler(config.hidden_size, pooling=pooling, norm_eps=config.norm_eps)

        elif 'muppet' in model:
            assert using_pretrained == True, "Only support pretrained muppet!"
            self.base_model = RobertaModel.from_pretrained(model)
            self.mlm_head = RobertaForMaskedLM(AutoConfig.from_pretrained(model)).lm_head

        elif 'roberta' in model:
            if using_pretrained:
                self.base_model = RobertaModel.from_pretrained(model)
                self.mlm_head = RobertaForMaskedLM.from_pretrained(model).lm_head
            else:
                self.base_model = RobertaModel(AutoConfig.from_pretrained(model))
                self.mlm_head = RobertaForMaskedLM(AutoConfig.from_pretrained(model)).lm_head

        elif 'albert' in model:
            if using_pretrained:
                self.base_model = AlbertModel.from_pretrained(model)
                self.mlm_head = AlbertForMaskedLM.from_pretrained(model).predictions
            else:
                self.base_model = AlbertModel(AutoConfig.from_pretrained(model))
                self.mlm_head = AlbertForMaskedLM(AutoConfig.from_pretrained(model)).predictions

        elif 'bert' in model:
            if using_pretrained:
                self.base_model = BertModel.from_pretrained(model)
                self.mlm_head = BertForPreTraining.from_pretrained(model).cls.predictions
            else:
                self.base_model = BertModel(AutoConfig.from_pretrained(model))
                self.mlm_head = BertForPreTraining(AutoConfig.from_pretrained(model)).cls.predictions

        else:
            raise ValueError(f"Unsupported backbone '{model}'")

        if gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()

        self.bin_layer = nn.Linear(self.base_model.config.hidden_size, 2)
        self.tri_layer = nn.Linear(self.base_model.config.hidden_size, 3)
        self.reg_layer = nn.Linear(self.base_model.config.hidden_size, 1)

        self.dropout = nn.Dropout(p=0.1)

        self.need_mlm = True
        self.is_finetune = False
        self.mlm_loss_factor = 0.5

        self.softmax = nn.Softmax(dim=-1)

        # names of the extra (held-out) validation sets, set by train.py; index 0 is the in-pool split
        self.val_set_names = []
        self._val_outputs = {}

        # from_pretrained() returns modules in eval mode and Lightning 2 preserves per-module modes in fit(),
        # which would silently disable dropout in the encoder for the whole run.
        self.train()

    def encode(self, batch):
        inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
        if 'token_type_ids' in batch and not self.is_modernbert:  # ModernBERT has no segment embeddings
            inputs['token_type_ids'] = batch['token_type_ids']
        output = self.base_model(**inputs)
        if self.pooler is not None:
            pooled = self.pooler(output.last_hidden_state, batch['attention_mask'])
        else:
            pooled = output.pooler_output
        return output, pooled

    def forward(self, batch):
        base_model_output, pooled_output = self.encode(batch)

        seq_relationship_score = self.bin_layer(self.dropout(pooled_output))  ## pooled output for classification
        tri_label_score = self.tri_layer(self.dropout(pooled_output))
        reg_label_score = self.reg_layer(pooled_output)

        prediction_scores = None
        losses, loss_nums = None, None
        if 'mlm_label' in batch.keys():  ### 'mlm_label' and 'align_label' when training
            ce_loss_fct = nn.CrossEntropyLoss(reduction='sum')
            mlm_label = batch['mlm_label']
            masked = mlm_label != -100
            masked_lm_loss_num = masked.sum()
            if self.need_mlm and masked_lm_loss_num > 0:
                # only project masked positions: a full (B x L x vocab) logit tensor at 8k tokens is huge
                prediction_scores = self.mlm_head(base_model_output.last_hidden_state[masked])
                masked_lm_loss = ce_loss_fct(prediction_scores.float(), mlm_label[masked])
            else:
                masked_lm_loss = seq_relationship_score.new_zeros(())

            next_sentence_loss = ce_loss_fct(seq_relationship_score.float().view(-1, 2), batch['align_label'].view(-1)) / math.log(2)
            tri_label_loss = ce_loss_fct(tri_label_score.float().view(-1, 3), batch['tri_label'].view(-1)) / math.log(3)
            reg_label_loss = self.mse_loss(reg_label_score.float().view(-1), batch['reg_label'].view(-1), reduction='sum')

            next_sentence_loss_num = torch.sum(batch['align_label'].view(-1) != -100)
            tri_label_loss_num = torch.sum(batch['tri_label'].view(-1) != -100)
            reg_label_loss_num = torch.sum(batch['reg_label'].view(-1) != -100.0)

            losses = [masked_lm_loss, next_sentence_loss, tri_label_loss, reg_label_loss]
            loss_nums = [masked_lm_loss_num, next_sentence_loss_num, tri_label_loss_num, reg_label_loss_num]

        return ModelOutput(
            loss=None,
            all_loss=losses,
            loss_nums=loss_nums,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            tri_label_logits=tri_label_score,
            reg_label_logits=reg_label_score,
            hidden_states=base_model_output.hidden_states,
            attentions=base_model_output.attentions
        )

    def _reduce_losses(self, losses, loss_nums):
        assert len(loss_nums) == len(losses), 'loss_num should be the same length as losses'
        reduced = []
        for loss, num in zip(losses, loss_nums):
            num = torch.sum(num)
            reduced.append(torch.sum(loss) / num if num > 0 else torch.sum(loss) * 0.)
        loss_mlm, loss_bin, loss_tri, loss_reg = reduced
        total_loss = self.mlm_loss_factor * loss_mlm + loss_bin + loss_tri + loss_reg
        return total_loss, loss_mlm, loss_bin, loss_tri, loss_reg

    def training_step(self, train_batch, batch_idx):
        output = self(train_batch)
        total_loss, loss_mlm, loss_bin, loss_tri, loss_reg = self._reduce_losses(output.all_loss, output.loss_nums)
        bs = train_batch['input_ids'].size(0)
        self.log('train_loss', total_loss, prog_bar=True, batch_size=bs)
        self.log('mlm_loss', loss_mlm, batch_size=bs)
        self.log('bin_label_loss', loss_bin, batch_size=bs)
        self.log('tri_label_loss', loss_tri, batch_size=bs)
        self.log('reg_label_loss', loss_reg, batch_size=bs)
        self.log('tokens_per_batch', train_batch['attention_mask'].sum().float(), batch_size=bs)
        return total_loss

    def on_validation_epoch_start(self):
        self._val_outputs = {}

    def validation_step(self, val_batch, batch_idx, dataloader_idx=0):
        output = self(val_batch)
        bs = val_batch['input_ids'].size(0)
        if dataloader_idx == 0:
            # in-pool held-out split: same multi-task loss as training
            total_loss, _, loss_bin, loss_tri, loss_reg = self._reduce_losses(output.all_loss, output.loss_nums)
            self.log('val_loss', total_loss, prog_bar=True, sync_dist=True, batch_size=bs, add_dataloader_idx=False)
            self.log('val_bin_loss', loss_bin, sync_dist=True, batch_size=bs, add_dataloader_idx=False)
            self.log('val_tri_loss', loss_tri, sync_dist=True, batch_size=bs, add_dataloader_idx=False)
            return

        # held-out (long-context) validation sets: collect alignment scores for AUC / balanced acc.
        # 3-way examples are scored with P(entail), binary ones with P(aligned) -- same as AlignScore 'nli'/'bin'.
        tri_p = self.softmax(output.tri_label_logits.float())[:, 0]
        bin_p = self.softmax(output.seq_relationship_logits.float())[:, 1]
        tri_label = val_batch['tri_label']
        is_tri = tri_label != -100
        score = torch.where(is_tri, tri_p, bin_p)
        label = torch.where(is_tri, (tri_label == 0).long(), val_batch['align_label'])
        keep = label != -100
        bucket = self._val_outputs.setdefault(dataloader_idx, {'score': [], 'label': []})
        bucket['score'].extend(score[keep].tolist())
        bucket['label'].extend(label[keep].tolist())

    def on_validation_epoch_end(self):
        if not self.val_set_names:
            return
        gathered = self._val_outputs
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            all_outputs = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(all_outputs, self._val_outputs)
            gathered = {}
            for rank_out in all_outputs:
                for idx, b in rank_out.items():
                    g = gathered.setdefault(idx, {'score': [], 'label': []})
                    g['score'].extend(b['score'])
                    g['label'].extend(b['label'])

        aucs = []
        for idx, b in sorted(gathered.items()):
            name = self.val_set_names[idx - 1] if idx - 1 < len(self.val_set_names) else f'val_{idx}'
            labels, scores = b['label'], b['score']
            if len(set(labels)) < 2:
                continue
            auc = roc_auc_score(labels, scores)
            preds = [int(s > 0.5) for s in scores]
            pos = [p for p, l in zip(preds, labels) if l == 1]
            neg = [1 - p for p, l in zip(preds, labels) if l == 0]
            bacc = 0.5 * (sum(pos) / len(pos) + sum(neg) / len(neg))
            self.log(f'val_auc/{name}', auc, rank_zero_only=True, add_dataloader_idx=False)
            self.log(f'val_bacc/{name}', bacc, rank_zero_only=True, add_dataloader_idx=False)
            aucs.append(auc)
        if aucs:
            self.log('val_auc_mean', sum(aucs) / len(aucs), prog_bar=True, rank_zero_only=True, add_dataloader_idx=False)
        self._val_outputs = {}

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight", "norm.weight"]  # ModernBERT norms are named *_norm / norm
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.hparams.warmup_steps_portion * self.trainer.estimated_stepping_batches),
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def mse_loss(self, input, target, ignored_index=-100.0, reduction='mean'):
        mask = (target == ignored_index)
        out = (input[~mask]-target[~mask])**2
        if reduction == "mean":
            return out.mean()
        elif reduction == "sum":
            return out.sum()


@dataclass
class ModelOutput():
    loss: Optional[torch.FloatTensor] = None
    all_loss: Optional[list] = None
    loss_nums: Optional[list] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    tri_label_logits: torch.FloatTensor = None
    reg_label_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
