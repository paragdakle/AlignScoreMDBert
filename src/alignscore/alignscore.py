from .inference import Inferencer
from typing import List, Optional

class AlignScore:
    def __init__(self, model: str, batch_size: int, device: int, ckpt_path: str, evaluation_mode='nli_sp', verbose=True,
                 max_length: Optional[int] = None, chunk_words: Optional[int] = None) -> None:
        """
        max_length:  max pair length in tokens (default: the backbone's limit, 512 for RoBERTa, 8192 for ModernBERT)
        chunk_words: context chunk size in words for the *_sp modes (default: 350 for 512-token models,
                     scaled to the window for long-context models)
        """
        self.model = Inferencer(
            ckpt_path=ckpt_path,
            model=model,
            batch_size=batch_size,
            device=device,
            verbose=verbose,
            max_length=max_length,
            chunk_words=chunk_words,
        )
        self.model.nlg_eval_mode = evaluation_mode

    def score(self, contexts: List[str], claims: List[str]) -> List[float]:
        return self.model.nlg_eval(contexts, claims)[1].tolist()
