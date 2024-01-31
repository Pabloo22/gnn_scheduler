"""Functions for creating features from the DenseData objects."""
import torch

from gnn_scheduler.gan.data import DenseData
from gnn_scheduler.job_shop.graphs import EdgeType


def get_normalized_duration_by_job(
    dense_data: DenseData, duration_column: int = 0
):
    """Returns an tensor of size (n_nodes, 1) containing the normalized
    duration of each job.

    Args:
        duration_column (int): the index of the duration column in the
            node features.
    """

    durations = dense_data.x[:, duration_column]
    conjunctive_adjacency_matrix = get_conjuctive_adjacency_matrix(dense_data)
    norm_durations = get_normalized_duration_by_job_from_conjunctive_matrix(
        conjunctive_adjacency_matrix, durations
    )
    return norm_durations


def get_normalized_duration_by_job_from_conjunctive_matrix(
    conjunctive_adjacency_matrix: torch.Tensor,
    durations: torch.Tensor,
):
    job_ids = get_job_ids(conjunctive_adjacency_matrix)
    


def get_job_ids(conjunctive_adjacency_matrix: torch.Tensor):
    n_nodes = len(conjunctive_adjacency_matrix)
    job_ids = [-1] * n_nodes
    n_jobs = 0
    for node_index in range(n_nodes):
        predecessor = _get_operation_predecessor(
            conjunctive_adjacency_matrix, node_index
        )

        has_no_predecessor = predecessor == -1
        if has_no_predecessor:
            job_ids[node_index] = n_jobs
            n_jobs += 1
            continue

        job_id, predecessors = _find_job_id_and_predecessors(
            conjunctive_adjacency_matrix, job_ids, node_index
        )
        nodes_to_set = predecessors
        nodes_to_set.append(node_index)
        for node_index in nodes_to_set:
            job_ids[node_index] = job_id


def _find_job_id_and_predecessors(
    conjunctive_adjacency_matrix: torch.Tensor,
    job_ids: list[int],
    node_index: int,
):
    predecessor = _get_operation_predecessor(
        conjunctive_adjacency_matrix, node_index
    )
    predecessors = [predecessor]
    predecessor_has_job_id = job_ids[predecessor] != -1
    while not predecessor_has_job_id:
        predecessor = _get_operation_predecessor(
            conjunctive_adjacency_matrix, predecessor
        )
        predecessors.append(predecessor)

        predecessor_has_job_id = job_ids[predecessor] != -1

    job_id = job_ids[predecessor]

    return job_id, predecessors


def _set_job_ids(
    job_ids: list[int],
    job_id: int,
    node_indices: list[int],
):
    for node_index in node_indices:
        job_ids[node_index] = job_id


def _assign_job_ids_to_first_operations(
    job_ids: list[int], conjunctive_adjacency_matrix: torch.Tensor
):
    first_operation_indices = torch.where(
        conjunctive_adjacency_matrix.sum(dim=0) == 0
    )[0]

    for job_id, node_index in enumerate(first_operation_indices):
        job_ids[node_index] = job_id


def _assign_job_ids_to_subsequent_nodes(
    job_ids: list[int], conjunctive_adjacency_matrix: torch.Tensor
):
    n_nodes = conjunctive_adjacency_matrix.size(0)


def _get_operation_predecessor(
    conjunctive_adjacency_matrix: torch.Tensor, node_index: int
) -> int:
    predecessor_index = torch.where(
        conjunctive_adjacency_matrix[:, node_index] == 1
    )[0]
    return predecessor_index if len(predecessor_index) > 0 else -1


def get_conjuctive_adjacency_matrix(
    dense_data: DenseData,
    conjunctive_adjacency_matrix_index: int = EdgeType.CONJUNCTIVE.value,
) -> torch.Tensor:
    conjunctive_adjacency_matrix = dense_data.adj_matrix[
        conjunctive_adjacency_matrix_index
    ]
    return conjunctive_adjacency_matrix
