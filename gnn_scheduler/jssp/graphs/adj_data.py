from typing import NamedTuple

import torch


class AdjData(NamedTuple):
    """Stores the adjacency matrix and the node features of a graph."""

    adj_matrix: torch.Tensor
    node_features: torch.Tensor

    def __repr__(self) -> str:
        return (
            f"AdjData(adj_matrix={self.adj_matrix.size()}, "
            f"node_features={self.node_features.size()})"
        )
