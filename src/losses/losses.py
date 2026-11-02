from torch.nn import functional as F

def autoregressive_cross_entropy(logits, labels):
    if logits.ndim != 3 or labels.shape != logits.shape[:2]: raise ValueError('Expected [B,T,V] logits and [B,T] labels.')
    return F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
