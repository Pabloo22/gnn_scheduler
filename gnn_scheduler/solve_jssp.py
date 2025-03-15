from collections.abc import Sequence
from typing import Any
import torch
import numpy as np
from job_shop_lib.graphs import build_resource_task_graph
from job_shop_lib.reinforcement_learning import (
    SingleJobShopGraphEnv,
    ResourceTaskGraphObservation,
    ResourceTaskGraphObservationDict,
)
from job_shop_lib.dispatching import (
    create_composite_operation_filter,
    ReadyOperationsFilterType,
)
from gnn_scheduler.data import JobShopData, DEFAULT_FEATURE_OBSERVERS_TYPES


def setup_environment(
    job_shop_instance,
) -> tuple[
    ResourceTaskGraphObservation[SingleJobShopGraphEnv],
    ResourceTaskGraphObservationDict,
    dict[str, Any],
]:
    """
    Set up the job shop environment with appropriate wrappers.

    Args:
        job_shop_instance: The job shop instance to solve

    Returns:
        wrapped_env, obs, info - The wrapped environment and initial
        observation
    """
    # Build a graph from the job shop instance
    graph = build_resource_task_graph(job_shop_instance)

    # Create an environment
    env = SingleJobShopGraphEnv(
        graph,
        feature_observer_configs=DEFAULT_FEATURE_OBSERVERS_TYPES,
        ready_operations_filter=create_composite_operation_filter(
            [
                ReadyOperationsFilterType.DOMINATED_OPERATIONS,
                ReadyOperationsFilterType.NON_IMMEDIATE_OPERATIONS,
            ]
        ),
    )

    # Wrap the environment with the observation wrapper
    wrapped_env = ResourceTaskGraphObservation(env)

    # Initialize the environment
    obs, info = wrapped_env.reset()

    return wrapped_env, obs, info


def normalize_features(
    node_features_dict: dict[str, np.ndarray],
    indices_to_normalize: dict[str, Sequence[int]] | None = None,
):
    """
    Normalize node features by dividing by the maximum values.

    Args:
        node_features_dict:
            Dictionary of node features
        indices_to_normalize:
            Dictionary specifying which indices to normalize for each node type

    Returns:
        Dictionary of normalized node features
    """
    if indices_to_normalize is None:
        indices_to_normalize = {
            "operation": list(range(8)),
            "machine": list(range(4)),
        }

    # Create a deep copy to avoid modifying the original
    normalized_dict = {}
    for key, features in node_features_dict.items():
        normalized_dict[key] = features.copy()

        if key in indices_to_normalize:
            # Divide by the maximum value checking for division by zero
            max_values = np.max(normalized_dict[key], axis=0)
            max_values[max_values == 0] = 1
            normalized_dict[key][:, indices_to_normalize[key]] /= max_values[
                indices_to_normalize[key]
            ]

    return normalized_dict


def create_job_shop_data(
    normalized_obs, edge_index_dict, available_ops, device
):
    """
    Create a JobShopData instance from observations for model input.

    Args:
        normalized_obs: Normalized node features dictionary
        edge_index_dict: Edge indices dictionary from observation
        available_ops: Available operations with their IDs
        device: The device to put tensors on

    Returns:
        JobShopData: Formatted data for model input
    """
    job_shop_data = JobShopData()

    # Add node features
    for node_type, features in normalized_obs.items():
        job_shop_data[node_type].x = torch.from_numpy(features).to(device)

    # Add edge indices
    for edge_type, indices in edge_index_dict.items():
        # Convert to torch.int64 (long) explicitly
        job_shop_data[edge_type].edge_index = torch.tensor(
            indices, device=device, dtype=torch.int64
        )
    # Add valid pairs
    job_shop_data["valid_pairs"] = torch.tensor(
        available_ops, device=device, dtype=torch.int64
    )

    return job_shop_data


def predict_best_action(model, job_shop_data) -> tuple[int, torch.Tensor]:
    """
    Use the GNN model to predict the best action.

    Args:
        model: The trained GNN model
        job_shop_data: Formatted job shop data

    Returns:
        best_action_idx, scores - The index of the best action and
        all scores
    """
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        scores = model(job_shop_data)

    # Choose the action with the highest score
    best_action_idx = torch.argmax(scores).item()

    return best_action_idx, scores  # type: ignore[return-value]


def map_to_original_action(
    action_tuple: tuple[int, int, int], original_ids_dict: dict[str, list[int]]
):
    """
    Map action tuple indices from zero-based to original IDs.

    Args:
        action_tuple:
            Tuple of (operation_id, machine_id, job_id)
        original_ids_dict:
            Dictionary mapping zero-based indices to original IDs

    Returns:
        tuple: The action tuple with original IDs
    """
    _, machine_id, job_id = action_tuple
    if "machine" in original_ids_dict:
        machine_id = original_ids_dict["machine"][machine_id]

    # Note: job_id typically doesn't need mapping as it's already in original
    # space
    return (job_id, machine_id)  # Return in format expected by env.step()


def solve_job_shop_with_gnn(job_shop_instance, model):
    """
    Solve a job shop scheduling instance using a GNN model to select actions.

    Args:
        job_shop_instance: The job shop instance to solve
        model: The trained GNN model for action selection

    Returns:
        Schedule: The final schedule produced by the solver
    """
    wrapped_env, obs, info = setup_environment(job_shop_instance)
    device = next(model.parameters()).device
    done = False
    while not done:
        available_ops = info["available_operations_with_ids"]
        if not available_ops:
            break
        normalized_obs = normalize_features(obs["node_features_dict"])
        job_shop_data = create_job_shop_data(
            normalized_obs, obs["edge_index_dict"], available_ops, device
        )
        if len(available_ops) == 1:
            best_action_idx = 0
        else:
            best_action_idx, _ = predict_best_action(model, job_shop_data)
        best_action_tuple = job_shop_data["valid_pairs"][
            best_action_idx
        ].tolist()
        action = map_to_original_action(
            best_action_tuple, obs["original_ids_dict"]
        )
        next_obs, _, done, _, info = wrapped_env.step(action)
        obs = next_obs

    schedule = wrapped_env.unwrapped.dispatcher.schedule

    return schedule
