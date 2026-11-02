import math
import torch

def build_scheduler(optimizer, total_steps, warmup_steps):
    def factor(step):
        if step < warmup_steps: return (step + 1) / max(1, warmup_steps)
        return .5 * (1 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
