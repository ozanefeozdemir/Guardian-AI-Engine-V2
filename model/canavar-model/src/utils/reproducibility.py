"""Seed setting and deterministic mode for reproducibility."""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_deterministic(enabled: bool = True):
    """Enable deterministic mode for PyTorch (may reduce performance)."""
    torch.backends.cudnn.deterministic = enabled
    torch.backends.cudnn.benchmark = not enabled
    if enabled:
        torch.use_deterministic_algorithms(True, warn_only=True)


def setup_reproducibility(config: dict):
    """Setup reproducibility from config."""
    seed = config.get('project', {}).get('seed', 42)
    set_seed(seed)
    set_deterministic(False)  # Keep benchmark=True for speed by default
