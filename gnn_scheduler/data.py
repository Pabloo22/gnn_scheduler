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
    InMemoryDataset,
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


class JobShopDataset(InMemoryDataset):

    def __init__(
        self,
        root: str = str(get_data_path()),
        transform=None,
        pre_transform=None,
        processed_filename: str = "job_shop_data.pt",
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
    ):
        self.feature_observers_types = (
            feature_observers_types
            if feature_observers_types is not None
            else _DEFAULT_FEATURE_OBSERVERS_TYPES
        )
        self.processed_filename = processed_filename
        self.raw_filename = raw_filename
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list[str]:
        return [os.path.join(self.raw_dir, self.raw_filename)]

    @property
    def processed_file_names(self) -> list[str]:
        return [os.path.join(self.processed_dir, self.processed_filename)]

    def download(self):
        if os.path.exists(self.raw_paths[0]):
            return
        url = (
            "https://github.com/Pabloo22/gnn_scheduler/blob/main/data/raw/"
            + self.raw_filename
        )
        download_url(url, self.raw_dir)

    def process(self):
        schedules_json = self._load_schedules()
        observations, action_probabilities_sequence = (
            self.get_observation_action_pairs(
                schedules_json, self.feature_observers_types
            )
        )
        # Save intermediate results
        with open(self.processed_dir + "/observations.pkl", "wb") as f:
            pickle.dump(observations, f)
        with open(
            self.processed_dir + "/action_probabilities.pkl",
            "wb",
        ) as f:
            pickle.dump(action_probabilities_sequence, f)

        hetero_dataset = self.process_observation_action_pairs(
            observations, action_probabilities_sequence
        )
        self.save(hetero_dataset, path=self.processed_paths[0])

    def _load_schedules(self):
        schedules_json = []
        for raw_path in self.raw_paths:
            with open(raw_path, "r", encoding="utf-8") as f:
                schedules_json.extend(json.load(f))
        return schedules_json[:100]

    @staticmethod
    def process_observation_action_pairs(
        observations: list[ResourceTaskGraphObservationDict],
        action_probabilities_sequence: list[dict[tuple[int, int, int], float]],
    ) -> list[JobShopData]:
        hetero_dataset: list[JobShopData] = []
        for obs, action_probs in tqdm.tqdm(
            zip(observations, action_probabilities_sequence),
            desc="Processing observation-action pairs",
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
