from typing import Any
from collections.abc import Iterator
import gc

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
        self._current_dataset: JobShopDataset | None = None
        self._current_dataloader: DataLoader | None = None

    def __len__(self):
        return len(self.raw_filenames)

    def __iter__(self) -> Iterator[DataLoader]:
        self._current_filename_index = 0
        self._cleanup_current_data()
        return self

    def _cleanup_current_data(self):
        """Explicitly clean up current dataset and dataloader to free
        memory."""
        if (
            hasattr(self, "_current_dataset")
            and self._current_dataset is not None
        ):
            del self._current_dataset
            self._current_dataset = None

        if (
            hasattr(self, "_current_dataloader")
            and self._current_dataloader is not None
        ):
            del self._current_dataloader
            self._current_dataloader = None

        # Force garbage collection to reclaim memory
        gc.collect()

    def __next__(self) -> DataLoader:
        if self._current_filename_index >= len(self.raw_filenames):
            raise StopIteration

        # Clean up previous dataset before loading a new one
        self._cleanup_current_data()

        raw_filename = self.raw_filenames[self._current_filename_index]
        self._current_filename_index += 1

        # Store references to current dataset and dataloader
        self._current_dataset = JobShopDataset(
            raw_filename=raw_filename, **self.dataset_kwargs
        )
        self._current_dataloader = DataLoader(
            self._current_dataset,
            **self.dataloader_kwargs,
        )

        return self._current_dataloader
