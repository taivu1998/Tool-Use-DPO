import os
import random
import logging
import torch
import numpy as np
from typing import Optional

def setup_logging(log_file: Optional[str] = None, level=logging.INFO):
    """Configures logging to console and optional file."""
    handlers = [logging.StreamHandler()]
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:  # Only create directory if path has a directory component
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=level,
        handlers=handlers
    )

def seed_everything(seed: int = 42):
    """Ensures reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logging.info(f"Global seed set to {seed}")

def get_device() -> str:
    """Returns the available computation device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"