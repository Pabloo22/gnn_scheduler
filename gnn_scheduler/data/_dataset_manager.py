from typing import Any
from collections.abc import Iterator

from torch_geometric.loader import DataLoader
from gnn_scheduler.data import JobShopDataset


class DatasetManager:
    """Manages the creation, loading, and unloading of data chunks for
    efficient training.

    Creates chunks lazily (on-demand) and manages them with a strict
    one-chunk-in-memory policy.
    """

    def __init__(
        self,
        raw_filenames: str | list[str] = "small_random_instances_0.json",
        dataset_kwargs: dict[str, Any] | None = None,
        dataloader_kwargs: dict[str, Any] | None = None,
    ):
        if isinstance(raw_filenames, str):
            raw_filenames = [raw_filenames]
        self.raw_filenames = raw_filenames
        self._current_filename_index = 0
        self.dataset_kwargs = dataset_kwargs or {}
        self.dataloader_kwargs = dataloader_kwargs or {}

    def __len__(self):
        return len(self.raw_filenames)

    def __iter__(self) -> Iterator[DataLoader]:
        self._current_filename_index = 0
        return self

    def __next__(self) -> DataLoader:
        if self._current_filename_index >= len(self.raw_filenames):
            raise StopIteration
        raw_filename = self.raw_filenames[self._current_filename_index]
        self._current_filename_index += 1
        dataset = JobShopDataset(
            raw_filename=raw_filename, **self.dataset_kwargs
        )
        dataloader = DataLoader(
            dataset,
            **self.dataloader_kwargs,
        )
        return dataloader
