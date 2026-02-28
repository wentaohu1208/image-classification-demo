"""Random seed management for reproducibility."""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: Optional[int] = 42) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value. If None, no seed is set.

    Example:
        >>> set_seed(42)
        >>> # All subsequent random operations will be deterministic
    """
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
