"""
Utilities: reproducibility, IO.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int):
    """Set seeds for Python random, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> str:
    """Create directory `path` (and parents) if missing. Returns `path`."""
    os.makedirs(path, exist_ok=True)
    return path
