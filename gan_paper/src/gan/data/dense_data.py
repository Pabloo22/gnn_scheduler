from typing import NamedTuple, Optional

import numpy as np
import torch

from gnn_scheduler.job_shop.graphs import EdgeType


class DenseData(NamedTuple):
    """Stores the adjacency matrix and the node features of a graph."""

    adj_matrix: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor | float

    def __repr__(self) -> str:
        target = self.y if isinstance(self.y, float) else self.y.size()
        return (
            f"DenseData(adj_matrix={self.adj_matrix.size()}, "
            f"x={self.x.size()}, "
            f"y={target})"
        )

    def get_tensors(
        self,
        device: torch.device,
        label_threshold: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the node features, adjacency matrices, and labels from a
        DenseData object.
        """
        node_features = self.x.to(device)

        if self.adj_matrix.is_sparse:
            adj_matrices = self.adj_matrix.to_dense().to(device)
        else:
            adj_matrices = self.adj_matrix.to(device)

        if label_threshold is not None:
            y = int(self.y >= label_threshold)
        else:
            y = self.y

        label = torch.tensor([y], dtype=torch.float).to(device)
        return node_features, adj_matrices, label

    def add_features(self, new_features: torch.Tensor):
        """Adds the given features to the node features of this DenseData
        object.
        """
        self.x = torch.cat([self.x, new_features], dim=1)

    def get_normalized_duration_by_job(self, duration_column: int = 0):
        """Returns an tensor of size (n_nodes, 1) containing the normalized
        duration of each job.

        Args:
            duration_column (int): the index of the duration column in the
                node features.
        """
        non_normalized_durations = self.x[:, duration_column]
        conjunctive_adjacency_matrix = self.get_conjuctive_adjacency_matrix()
        disjunctive_adjacency_matrix = self.get_disjunction_adjacency_matrix()
        job_ids = self.get_job_ids()

    def get_conjuctive_adjacency_matrix(
        self,
        conjunctive_adjacency_matrix_index: int = EdgeType.CONJUNCTIVE.value,
    ) -> torch.Tensor:
        conjunctive_adjacency_matrix = self.adj_matrix[
            conjunctive_adjacency_matrix_index
        ]
        return conjunctive_adjacency_matrix

    def get_disjunction_adjacency_matrix(
        self,
        disjunction_adjacency_matrix_index: int = EdgeType.DISJUNCTIVE.value,
    ) -> torch.Tensor:
        disjunction_adjacency_matrix = self.adj_matrix[
            disjunction_adjacency_matrix_index
        ]
        return disjunction_adjacency_matrix

    @classmethod
    def get_job_ids(cls, conjunctive_adjacency_matrix: torch.Tensor):
        n_nodes = conjunctive_adjacency_matrix.size(0)
        job_ids = [-1] * n_nodes

        cls._assign_job_ids_to_first_operations(
            job_ids, conjunctive_adjacency_matrix
        )

        # Now, we need to find the nodes that have incoming edges from
        # the first operation of a job (i.e. the second operation of each job)
        # and assign a job id to these nodes, and so on.

    @staticmethod
    def _assign_job_ids_to_first_operations(
        job_ids: list[int], conjunctive_adjacency_matrix: torch.Tensor
    ):
        first_operation_indices = torch.where(
            conjunctive_adjacency_matrix.sum(dim=0) == 0
        )[0]

        for job_id, node_index in enumerate(first_operation_indices):
            job_ids[node_index] = job_id

    @staticmethod
    def _assign_job_ids_to_subsequent_nodes(
        job_ids: list[int], conjunctive_adjacency_matrix: torch.Tensor
    ):
        n_nodes = conjunctive_adjacency_matrix.size(0)

        # Traverse the adjacency matrix to assign job IDs to subsequent nodes
        for _ in range(n_nodes):
            for node_index in range(n_nodes):
                # Skip if job_id is already assigned
                if job_ids[node_index] != -1:
                    continue

                # Check if all predecessors have a job id assigned
                predecessors = conjunctive_adjacency_matrix[:, node_index]
                predecessor_indices = torch.where(predecessors == 1)[0]
                if all(job_ids[pred] != -1 for pred in predecessor_indices):
                    # Assign job_id based on the first predecessor
                    job_ids[node_index] = job_ids[predecessor_indices[0]]
