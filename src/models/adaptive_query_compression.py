"""Question-aware adaptive query selection, retaining the fixed DocVLM path unchanged."""
import torch
from torch import nn

class QuestionAwareQuerySelector(nn.Module):
    """Scores candidate OCR-query states against the encoded question and selects top-K.

    Selection is discrete at inference. During training the selected scores weight states,
    allowing gradients through the scorer and selected candidates.
    """
    def __init__(self, hidden_size: int, max_queries: int, adaptive_k: int):
        super().__init__(); self.max_queries=max_queries; self.adaptive_k=adaptive_k
        self.question_projection=nn.Linear(hidden_size,hidden_size,bias=False); self.query_projection=nn.Linear(hidden_size,hidden_size,bias=False); self.score_bias=nn.Parameter(torch.zeros(max_queries))
    def forward(self, candidates: torch.Tensor, question_features: torch.Tensor):
        # ``question_features`` is [B, T, H]; a row is padding only when every feature is zero.
        qmask=question_features.abs().sum(-1).ne(0).unsqueeze(-1)
        question=(question_features*qmask).sum(1)/qmask.sum(1).clamp_min(1)
        scores=(self.query_projection(candidates)*self.question_projection(question).unsqueeze(1)).sum(-1)/candidates.size(-1)**.5 + self.score_bias[:candidates.size(1)]
        k=min(self.adaptive_k,candidates.size(1)); values,indices=scores.topk(k,dim=1); selected=candidates.gather(1,indices.unsqueeze(-1).expand(-1,-1,candidates.size(-1)))
        return selected*values.sigmoid().unsqueeze(-1),scores,indices
