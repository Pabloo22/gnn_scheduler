import os
import json
import sys

from typing import Type
from collections.abc import Sequence
from torch_geometric.data import (  # type: ignore[import-untyped]
    InMemoryDataset,
    download_url,
)
from torch_geometric.data.dataset import (  # type: ignore[import-untyped]
    files_exist,
)

from job_shop_lib.dispatching.feature_observers import (
    FeatureObserverType,
    FeatureObserver,
    FeatureObserverConfig,
)

from gnn_scheduler.data import (
    JobShopData,
    DEFAULT_FEATURE_OBSERVERS_TYPES,
    get_observation_action_pairs,
    process_observation_action_pairs,
)
from gnn_scheduler.utils import get_data_path


class JobShopDataset(InMemoryDataset):

    def __init__(
        self,
        root: str = str(get_data_path()),
        transform=None,
        pre_transform=None,
        raw_filename: str = "small_random_instances_0.json",
        feature_observers_types: (
            Sequence[
                str
                | FeatureObserverType
                | Type[FeatureObserver]
                | FeatureObserverConfig
            ]
            | None
        ) = None,
        force_reload: bool = False,
        log: bool = True,
        num_threads: int = -1,
    ):
        self.feature_observers_types = (
            feature_observers_types
            if feature_observers_types is not None
            else DEFAULT_FEATURE_OBSERVERS_TYPES
        )
        self.raw_filename = raw_filename

        cpu_count = os.cpu_count()
        if cpu_count is not None and num_threads == -1:
            num_threads = cpu_count - 1
        elif num_threads <= 0:
            num_threads = 1
        self.num_threads = num_threads
        super().__init__(
            root, transform, pre_transform, force_reload=force_reload, log=log
        )
        self.load(self.processed_paths[0], data_cls=JobShopData)

    @property
    def raw_file_names(self) -> list[str]:
        return [self.raw_filename]

    @property
    def processed_file_names(self) -> list[str]:
        raw_filename_stem = os.path.splitext(self.raw_filename)[0]
        return [f"{raw_filename_stem}_processed.pt"]

    def download(self):
        if os.path.exists(self.raw_paths[0]):
            return

        for raw_filename in self.raw_file_names:
            url = (
                "https://raw.githubusercontent.com/Pabloo22/gnn_scheduler/main"
                "/data/raw/small_random_instances_0.json/" + raw_filename
            )
            download_url(url, self.raw_dir)

    def process(self):
        # Create chunks for processing
        schedules_json = self._load_schedules()
        observation_action_pairs = get_observation_action_pairs(
            schedules_json, self.feature_observers_types
        )
        processed_data = process_observation_action_pairs(
            *observation_action_pairs
        )
        self.save(processed_data, self.processed_paths[0])

    def _load_schedules(self):
        schedules_json = []
        for raw_path in self.raw_paths:
            with open(raw_path, "r", encoding="utf-8") as f:
                schedules_json.extend(json.load(f))
        return schedules_json

    def _process(self):
        if not self.force_reload and files_exist(self.processed_paths):
            return

        if self.log and "pytest" not in sys.modules:
            print("Processing...", file=sys.stderr)
            print(
                "Reading the following files: "
                f"{', '.join(self.raw_file_names)}",
                file=sys.stderr,
            )

        os.makedirs(self.processed_dir, exist_ok=True)
        self.process()

        # No saving of pre_transform.pt and pre_filter.pt files here

        if self.log and "pytest" not in sys.modules:
            print("Done!", file=sys.stderr)
