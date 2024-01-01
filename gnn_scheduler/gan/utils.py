"""Contains utility functions for the difficulty prediction task."""
import random

import torch


def random_name() -> str:
    """Generates a random name for a run.

    Returns:
        str: The random name.
    """
    return "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=5))


def set_all_seeds(seed: int):
    """Sets all seeds to the given value.

    Args:
        seed (int): The seed to set.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
