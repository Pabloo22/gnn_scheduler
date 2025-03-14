import random
import threading
from typing import Type
from collections.abc import Sequence
import numpy as np
import torch
import tqdm

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

from gnn_scheduler.data import JobShopData


DEFAULT_FEATURE_OBSERVERS_TYPES = [
    FeatureObserverType.DURATION,
    FeatureObserverType.EARLIEST_START_TIME,
    FeatureObserverType.IS_SCHEDULED,
    FeatureObserverType.POSITION_IN_JOB,
    FeatureObserverType.REMAINING_OPERATIONS,
]


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
        job_shop_data = process_observation_action_pair(obs, action_probs)
        hetero_dataset.append(job_shop_data)
    return hetero_dataset


def process_observation_action_pair(
    observation: ResourceTaskGraphObservationDict,
    action_probabilities: dict[tuple[int, int, int], float],
) -> JobShopData:
    job_shop_data = JobShopData()
    for key, value in observation.items():
        for (
            subkey,
            subvalue,
        ) in value.items():  # type: ignore[attr-defined]
            if key == "node_features_dict":
                job_shop_data[subkey].x = torch.from_numpy(subvalue)
            elif key == "edge_index_dict":
                job_shop_data[subkey].edge_index = torch.from_numpy(subvalue)
    job_shop_data["y"] = torch.tensor(
        list(action_probabilities.values()), dtype=torch.float32
    )
    job_shop_data["valid_pairs"] = torch.tensor(
        list(action_probabilities.keys())
    )
    return job_shop_data


def map_available_ops_ids_to_original(
    available_operations_with_ids: list[tuple[int, int, int]],
    original_ids: dict[str, np.ndarray],
) -> list[tuple[int, int, int]]:
    new_ids = []
    for operation_id, machine_id, job_id in available_operations_with_ids:
        original_operation_id = original_ids["operation"][operation_id]
        original_machine_id = original_ids["machine"][machine_id]
        new_ids.append((original_operation_id, original_machine_id, job_id))
    return new_ids


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


def get_observation_action_pairs_with_threading(
    schedules_json: list[dict],
    feature_observers_types: Sequence[
        str
        | FeatureObserverType
        | Type[FeatureObserver]
        | FeatureObserverConfig,
    ],
    num_threads: int = 8,
) -> tuple[
    list[ResourceTaskGraphObservationDict],
    list[dict[tuple[int, int, int], float]],
]:
    """
    Process schedules to get observation-action pairs using multiple threads.

    Args:
        schedules_json: List of schedule dictionaries to process
        feature_observers_types: Feature observers to use for the environment
        num_threads: Number of threads to use for processing

    Returns:
        Tuple of (observations, action_probabilities_sequence)
    """
    if not schedules_json:
        return [], []

    # Split schedules into chunks for each thread
    chunk_size = max(1, len(schedules_json) // num_threads)
    schedule_chunks = [
        schedules_json[i : i + chunk_size]
        for i in range(0, len(schedules_json), chunk_size)
    ]

    # Limit number of threads to the number of chunks
    actual_num_threads = min(num_threads, len(schedule_chunks))

    # Results container for each thread
    thread_results: list[
        None
        | tuple[
            list[ResourceTaskGraphObservationDict],
            list[dict[tuple[int, int, int], float]],
        ]
    ] = [None] * actual_num_threads

    def process_chunk(thread_id: int, chunk: list[dict]):
        """Worker function for each thread to process its chunk of schedules"""
        result = get_observation_action_pairs(chunk, feature_observers_types)
        thread_results[thread_id] = result

    # Create and start threads
    threads: list[threading.Thread] = []
    for i in range(actual_num_threads):
        thread = threading.Thread(
            target=process_chunk,
            args=(i, schedule_chunks[i]),
        )
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Combine results from all threads
    all_observations: list[ResourceTaskGraphObservationDict] = []
    all_action_probabilities: list[dict[tuple[int, int, int], float]] = []
    for obs, action_probs in thread_results:  # type: ignore[misc]
        all_observations.extend(obs)
        all_action_probabilities.extend(action_probs)

    return all_observations, all_action_probabilities


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
    action_probabilities_sequence: list[dict[tuple[int, int, int], float]] = []
    for schedule_dict in tqdm.tqdm(
        schedules_json, desc="Processing schedules"
    ):
        schedule = Schedule.from_dict(**schedule_dict)
        obs, action_probs = get_observation_action_pairs_from_schedule(
            schedule, feature_observers_types
        )
        observations.extend(obs)
        action_probabilities_sequence.extend(action_probs)
    return observations, action_probabilities_sequence


def get_observation_action_pairs_from_schedule(
    schedule: Schedule,
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
    action_probabilities_sequence: list[dict[tuple[int, int, int], float]] = []
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
            map_available_ops_ids_to_original(
                info["available_operations_with_ids"],
                obs["original_ids_dict"],
            ),
        )
        if len(action_probs) > 1:
            obs["node_features_dict"] = normalize_features(
                obs["node_features_dict"]
            )
            observations.append(obs)
            action_probs_adjusted: dict[tuple[int, int, int], float] = {}
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
            action for action, value in action_probs.items() if value == 1.0
        ]
        action_choice = random.choice(optimal_actions)
        _, machine_id, job_id = action_choice
        obs, _, done, _, info = wrapped_env.step((job_id, machine_id))
    return observations, action_probabilities_sequence
