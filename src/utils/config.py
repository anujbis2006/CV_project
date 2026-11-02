from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML with one optional relative ``_base`` file."""
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    base = data.pop("_base", None)
    if base:
        parent = load_config(os.path.join(os.path.dirname(path), base))
        parent.update(data)
        return parent
    return data


@dataclass
class Config:
    seed: int = 42
    device: str = "auto"
    model_name: str = "docvlm-lite"
    batch_size: int = 8
    num_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    scheduler: str = "linear"
    gradient_accumulation_steps: int = 1
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    data_dir: str = "data"
    log_dir: str = "outputs/logs"
    image_size: int = 224
    max_seq_length: int = 128
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    save_every: int = 5
    eval_every: int = 1
    resume: Optional[str] = None
    debug: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        config = cls(**data)
        config.extra = {k: v for k, v in data.items() if k not in cls.__dataclass_fields__}
        return config

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


def get_default_config_path() -> str:
    return os.path.join(os.getcwd(), "configs", "debug.yaml")
