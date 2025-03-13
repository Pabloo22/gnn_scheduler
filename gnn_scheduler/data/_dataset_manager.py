from typing import Type
from collections.abc import Sequence, Iterator

from job_shop_lib.dispatching.feature_observers import (
    FeatureObserverType,
    FeatureObserver,
    FeatureObserverConfig,
)
from gnn_scheduler.utils import get_data_path
from gnn_scheduler.data import JobShopDataset

from job_shop_lib.generation import GeneralInstanceGenerator


class DatasetManager:
    """Manages the creation, loading, and unloading of data chunks for
    efficient training.

    Creates chunks lazily (on-demand) and manages them with a strict
    one-chunk-in-memory policy.
    """

    def __init__(
        self,
        root: str = str(get_data_path()),
        transform=None,
        pre_transform=None,
        raw_filenames: str | list[str] = ["small_random_instances_0.json"],
        feature_observers_types: (
            Sequence[
                str
                | FeatureObserverType
                | Type[FeatureObserver]
                | FeatureObserverConfig
            ]
            | None
        ) = None,
        num_chunks: int = 100,
        force_reload: bool = False,
        max_chunks_in_memory: int = 100,
        log: bool = True,
    ):
        if isinstance(raw_filenames, str):
            raw_filenames = [raw_filenames]
        self.raw_filenames = raw_filenames
        self._current_filename_index = 0
        # save attributes for reloading
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.feature_observers_types = feature_observers_types
        self.num_chunks = num_chunks
        self.force_reload = force_reload
        self.max_chunks_in_memory = max_chunks_in_memory
        self.log = log

    def __len__(self):
        return len(self.raw_filenames)

    def __iter__(self) -> Iterator[JobShopDataset]:
        self._current_filename_index = 0
        return self

    def __next__(self) -> JobShopDataset:
        if self._current_filename_index >= len(self.raw_filenames):
            raise StopIteration
        raw_filename = self.raw_filenames[self._current_filename_index]
        self._current_filename_index += 1
        return JobShopDataset(
            root=self.root,
            raw_filenames=raw_filename,
            transform=self.transform,
            pre_transform=self.pre_transform,
            feature_observers_types=self.feature_observers_types,
            num_chunks=self.num_chunks,
            force_reload=self.force_reload,
            max_chunks_in_memory=self.max_chunks_in_memory,
            log=self.log,
        )
