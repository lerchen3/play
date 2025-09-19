from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
from transformers import AutoTokenizer


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True


def load_tokenizer(tokenizer_path: str, pad_token: str = "[PAD]", **kwargs):
    """Load a tokenizer and ensure a pad token exists."""
    kwargs.setdefault('trust_remote_code', True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **kwargs)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': pad_token})
    return tokenizer


def flatten_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a nested dictionary by dropping intermediate keys."""
    flat: Dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, dict):
            flat.update(flatten_config(value))
        else:
            flat[key] = value
    return flat


def ensure_dir(path: Optional[str]) -> None:
    """Create a directory if the path is provided and does not exist."""
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def ensure_parent_dir(path: Optional[str]) -> None:
    """Create parent directories for a file path if needed."""
    if not path:
        return
    directory = path if os.path.isdir(path) else os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
