from __future__ import annotations

import os
from typing import Optional


def get_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def get_torch_device(requested: str = "auto"):
    import torch
    if requested != "auto": return torch.device(requested)
    return torch.device(get_device())


def get_cuda_visible_devices() -> Optional[str]:
    return os.environ.get("CUDA_VISIBLE_DEVICES")
