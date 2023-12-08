from typing import NamedTuple

import torch


class AdjData(NamedTuple):
    """Stores the adjacency matrix and the node features of a graph."""

    adj_matrix: torch.Tensor
    node_features: torch.Tensor
    target: torch.Tensor | float

    def __repr__(self) -> str:
        target = (
            self.target
            if isinstance(self.target, float)
            else self.target.size()
        )
        return (
            f"AdjData(adj_matrix={self.adj_matrix.size()}, "
            f"node_features={self.node_features.size()}, "
            f"target={target})"
        )
