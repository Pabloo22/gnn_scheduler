import os
import json
import random
import pickle
import sys

from typing import Type
from collections.abc import Sequence
import numpy as np
import torch
from torch_geometric.data import (  # type: ignore[import-untyped]
    HeteroData,
    Dataset,
    download_url,
)
from torch_geometric.data.dataset import files_exist
from torch_geometric.io import fs
import tqdm  # type: ignore[import-untyped]

from job_shop_lib.dispatching.feature_observers import (
    FeatureObserverType,
    FeatureObserver,
    FeatureObserverConfig,
)
from job_shop_lib.graphs import build_resource_task_graph
from job_shop_lib.reinforcement_learning import (
    SingleJobShopGraphEnv,
    ResourceTaskGraphObservation,
    get_optimal_actions,
    ResourceTaskGraphObservationDict,
)
from job_shop_lib.dispatching import OptimalOperationsObserver
from job_shop_lib import Schedule

from gnn_scheduler.utils import get_data_path


_DEFAULT_FEATURE_OBSERVERS_TYPES = [
    FeatureObserverType.DURATION,
    FeatureObserverType.EARLIEST_START_TIME,
    FeatureObserverType.IS_SCHEDULED,
    FeatureObserverType.POSITION_IN_JOB,
    FeatureObserverType.REMAINING_OPERATIONS,
]


class JobShopData(HeteroData):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "valid_pairs":
            assert isinstance(value, torch.Tensor)
            dim_size = value.size(1)
            increments = torch.zeros(
                dim_size, dtype=torch.long, device=value.device
            )

            # Set increments for each column according to respective node
            # counts
            node_types = ["operation", "machine", "job"]
            for i, node_type in enumerate(node_types):
                if self[node_type]:
                    increments[i] = self[node_type]["x"].size(0)

            return increments

        return super().__inc__(key, value, *args, **kwargs)


class JobShopDataset(Dataset):

    def __init__(
        self,
        root: str = str(get_data_path()),
        transform=None,
        pre_transform=None,
        processed_filenames_prefix: str = "job_shop_data",
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
        num_chunks: int = 1,
        force_reload: bool = False,
        max_chunks_in_memory: int = 2,
    ):
        self.feature_observers_types = (
            feature_observers_types
            if feature_observers_types is not None
            else _DEFAULT_FEATURE_OBSERVERS_TYPES
        )
        self.processed_filenames_prefix = processed_filenames_prefix
        self.raw_filename = raw_filename
        self.num_chunks = num_chunks
        self.max_chunks_in_memory = max_chunks_in_memory
        self._data_chunks: dict[int, list[JobShopData]] = {}
        self._chunk_access_order: list[int] = []  # Track LRU chunks
        self._chunk_sizes: list[int] = []
        self._total_size = 0
        self._metadata_file = "metadata.json"
        super().__init__(
            root, transform, pre_transform, force_reload=force_reload
        )
        self._load_metadata()

    @property
    def raw_file_names(self) -> list[str]:
        return [os.path.join(self.raw_dir, self.raw_filename)]

    @property
    def processed_file_names(self) -> list[str]:
        # Include metadata file and all chunk files
        files = [os.path.join(self.processed_dir, self._metadata_file)]
        files.extend(
            [
                os.path.join(
                    self.processed_dir,
                    f"{self.processed_filenames_prefix}_{i}.pt",
                )
                for i in range(self.num_chunks)
            ]
        )
        return files

    def download(self):
        if os.path.exists(self.raw_paths[0]):
            return
        url = (
            "https://github.com/Pabloo22/gnn_scheduler/blob/main/data/raw/"
            + self.raw_filename
        )
        download_url(url, self.raw_dir)

    def _get_chunk_path(self, chunk_idx: int) -> str:
        return os.path.join(
            self.processed_dir,
            f"{self.processed_filenames_prefix}_{chunk_idx}.pt",
        )

    def _load_metadata(self):
        metadata_path = os.path.join(self.processed_dir, self._metadata_file)
        if not os.path.exists(metadata_path):
            return

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                self._chunk_sizes = metadata.get("chunk_sizes", [])
                self._total_size = metadata.get("total_size", 0)
        except (json.JSONDecodeError, FileNotFoundError):
            # If metadata file doesn't exist or is invalid, we'll recreate it
            self._chunk_sizes = []
            self._total_size = 0

    def _save_metadata(self):
        metadata_path = os.path.join(self.processed_dir, self._metadata_file)
        metadata = {
            "chunk_sizes": self._chunk_sizes,
            "total_size": self._total_size,
            "num_chunks": self.num_chunks,
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)

    def process(self):
        # Create chunks for processing
        schedules_json = self._load_schedules()
        chunk_size = max(1, len(schedules_json) // self.num_chunks)

        self._chunk_sizes = []
        self._total_size = 0

        for chunk_idx in range(self.num_chunks):
            # check if file already exists
            chunk_path = self._get_chunk_path(chunk_idx)
            if not self.force_reload and os.path.exists(chunk_path):
                # Load chunk to update metadata
                hetero_dataset = torch.load(chunk_path)
                self._chunk_sizes.append(len(hetero_dataset))
                self._total_size += len(hetero_dataset)
                continue
            start_idx = chunk_idx * chunk_size
            end_idx = (
                min((chunk_idx + 1) * chunk_size, len(schedules_json))
                if chunk_idx < self.num_chunks - 1
                else len(schedules_json)
            )

            if start_idx >= end_idx:
                # No more data to process
                break

            # Process this chunk
            chunk_schedules = schedules_json[start_idx:end_idx]

            if self.log:
                print(
                    f"Processing chunk {chunk_idx+1}/{self.num_chunks} "
                    f"({len(chunk_schedules)} schedules)..."
                )

            # Process this chunk of schedules
            observations, action_probabilities_sequence = (
                self.get_observation_action_pairs(
                    chunk_schedules, self.feature_observers_types
                )
            )

            # Process observations and action probabilities into JobShopData
            # objects
            hetero_dataset = self.process_observation_action_pairs(
                observations, action_probabilities_sequence
            )

            # Save this chunk
            self._chunk_sizes.append(len(hetero_dataset))
            self._total_size += len(hetero_dataset)
            torch.save(hetero_dataset, chunk_path)

            if self.log:
                print(
                    f"Saved chunk {chunk_idx+1} with {len(hetero_dataset)} "
                    "samples"
                )

        # Save metadata
        self._save_metadata()

    def _load_schedules(self):
        schedules_json = []
        for raw_path in self.raw_paths:
            with open(raw_path, "r", encoding="utf-8") as f:
                schedules_json.extend(json.load(f))
        return schedules_json

    def len(self) -> int:
        """Returns the total number of samples across all chunks."""
        return self._total_size

    def get(self, idx: int) -> JobShopData:
        """Gets the data object at the specified index with memory management."""
        if idx < 0 or idx >= self.len():
            raise IndexError(
                f"Index {idx} out of range for dataset with {self.len()} "
                "samples"
            )

        # Find which chunk contains this index
        chunk_idx = 0
        sample_idx = idx

        for size in self._chunk_sizes:
            if sample_idx < size:
                break
            sample_idx -= size
            chunk_idx += 1

        # Load the chunk if not already loaded
        if chunk_idx not in self._data_chunks:
            # Check if we need to free memory first
            if len(self._data_chunks) >= self.max_chunks_in_memory:
                # Remove least recently used chunk
                lru_chunk = self._chunk_access_order.pop(0)
                del self._data_chunks[lru_chunk]
                if self.log:
                    print(f"Unloaded chunk {lru_chunk} to free memory")

            # Load the requested chunk
            chunk_path = self._get_chunk_path(chunk_idx)
            self._data_chunks[chunk_idx] = torch.load(chunk_path)
            if self.log:
                print(f"Loaded chunk {chunk_idx} into memory")
        elif chunk_idx in self._chunk_access_order:
            # Move this chunk to the end of the access order (most recently used)
            self._chunk_access_order.remove(chunk_idx)

        # Add/Update this chunk as most recently used
        self._chunk_access_order.append(chunk_idx)

        # Get the sample from the chunk
        data = self._data_chunks[chunk_idx][sample_idx]

        if self.transform is not None:
            data = self.transform(data)

        return data

    @staticmethod
    def process_observation_action_pairs(
        observations: list[ResourceTaskGraphObservationDict],
        action_probabilities_sequence: list[dict[tuple[int, int, int], float]],
    ) -> list[JobShopData]:
        hetero_dataset: list[JobShopData] = []
        assert len(observations) == len(action_probabilities_sequence)
        for obs, action_probs in tqdm.tqdm(
            zip(observations, action_probabilities_sequence),
            desc="Processing observation-action pairs",
            total=len(observations),
        ):
            job_shop_data = JobShopData()
            for key, value in obs.items():
                for subkey, subvalue in value.items():
                    if key == "node_features_dict":
                        job_shop_data[subkey].x = torch.from_numpy(subvalue)
                    elif key == "edge_index_dict":
                        job_shop_data[subkey].edge_index = torch.from_numpy(
                            subvalue
                        )
            job_shop_data["y"] = torch.tensor(
                list(action_probs.values()), dtype=torch.float32
            )
            job_shop_data["valid_pairs"] = torch.tensor(
                list(action_probs.keys())
            )
            hetero_dataset.append(job_shop_data)
        return hetero_dataset

    @staticmethod
    def get_observation_action_pairs(
        schedules_json: list[dict],
        feature_observers_types: Sequence[
            str
            | FeatureObserverType
            | Type[FeatureObserver]
            | FeatureObserverConfig,
        ],
    ) -> tuple[
        list[ResourceTaskGraphObservationDict],
        list[dict[tuple[int, int, int], float]],
    ]:
        observations: list[ResourceTaskGraphObservationDict] = []
        action_probabilities_sequence: list[
            dict[tuple[int, int, int], float]
        ] = []
        for schedule_dict in tqdm.tqdm(
            schedules_json, desc="Processing schedules"
        ):
            schedule = Schedule.from_dict(**schedule_dict)
            graph = build_resource_task_graph(schedule.instance)
            env = SingleJobShopGraphEnv(
                graph,
                feature_observer_configs=feature_observers_types,
                ready_operations_filter=None,
            )
            wrapped_env = ResourceTaskGraphObservation(env)
            optimal_ops_observer = OptimalOperationsObserver(
                wrapped_env.unwrapped.dispatcher, schedule
            )
            obs, info = wrapped_env.reset()
            done = False
            while not done:

                action_probs = get_optimal_actions(
                    optimal_ops_observer,
                    JobShopDataset.map_available_ops_ids_to_original(
                        info["available_operations_with_ids"],
                        obs["original_ids_dict"],
                    ),
                )
                if len(action_probs) > 1:
                    obs["node_features_dict"] = (
                        JobShopDataset.normalize_features(
                            obs["node_features_dict"]
                        )
                    )
                    observations.append(obs)
                    action_probs_adjusted: dict[
                        tuple[int, int, int], float
                    ] = {}
                    assert len(info["available_operations_with_ids"]) == len(
                        action_probs
                    )
                    for key, value in zip(
                        info["available_operations_with_ids"],
                        action_probs.values(),
                    ):
                        action_probs_adjusted[key] = value
                    action_probabilities_sequence.append(action_probs_adjusted)
                optimal_actions = [
                    action
                    for action, value in action_probs.items()
                    if value == 1.0
                ]
                action_choice = random.choice(optimal_actions)
                _, machine_id, job_id = action_choice
                obs, _, done, _, info = wrapped_env.step((job_id, machine_id))
        return observations, action_probabilities_sequence

    @staticmethod
    def map_available_ops_ids_to_original(
        available_operations_with_ids: list[tuple[int, int, int]],
        original_ids: dict[str, np.ndarray],
    ) -> list[tuple[int, int, int]]:
        new_ids = []
        for operation_id, machine_id, job_id in available_operations_with_ids:
            original_operation_id = original_ids["operation"][operation_id]
            original_machine_id = original_ids["machine"][machine_id]
            new_ids.append(
                (original_operation_id, original_machine_id, job_id)
            )
        return new_ids

    @staticmethod
    def normalize_features(
        node_features_dict: dict[str, np.ndarray],
        indices_to_normalize: dict[str, Sequence[int]] | None = None,
    ) -> dict[str, np.ndarray]:
        if indices_to_normalize is None:
            indices_to_normalize = {
                "operation": list(range(8)),
                "machine": list(range(4)),
            }
        for key, indices in indices_to_normalize.items():
            # Divide by the maximum value checking for division by zero
            max_values = np.max(node_features_dict[key], axis=0)
            max_values[max_values == 0] = 1
            node_features_dict[key][:, indices] /= max_values[indices]

        return node_features_dict

    def _process(self):
        if not self.force_reload and files_exist(self.processed_paths):
            return

        if self.log and "pytest" not in sys.modules:
            print("Processing...", file=sys.stderr)

        fs.makedirs(self.processed_dir, exist_ok=True)
        self.process()

        # No saving of pre_transform.pt and pre_filter.pt files here

        if self.log and "pytest" not in sys.modules:
            print("Done!", file=sys.stderr)
