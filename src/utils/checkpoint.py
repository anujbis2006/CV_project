from __future__ import annotations

import os
from typing import Any, Dict, Optional


def save_checkpoint(model: Any, optimizer: Any, scheduler: Any, path: str, epoch: int, metrics: Optional[Dict[str, float]] = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "metrics": metrics or {},
    }
    import torch

    torch.save(checkpoint, path)


def load_checkpoint(model: Any, path: str, optimizer: Optional[Any] = None, scheduler: Optional[Any] = None) -> Dict[str, Any]:
    import torch

    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    return checkpoint
