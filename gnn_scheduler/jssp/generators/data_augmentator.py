from __future__ import annotations

import random
from typing import Callable

from gnn_scheduler.jssp import JobShopInstance


Transformation = Callable[[JobShopInstance], JobShopInstance]


class DataAugmentator:
    """Applies a series of transformations to a JobShopInstance."""

    def __init__(self, transformations: list[Transformation], seed: int = 0):
        self.transformations = transformations
        self.seed = seed
        random.seed(seed)

    def generate(self, instance: JobShopInstance) -> JobShopInstance:
        new_instance = instance
        for transform in self.transformations:
            new_instance = transform.apply(new_instance)
        return new_instance
