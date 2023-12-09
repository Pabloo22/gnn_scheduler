from typing import NamedTuple

import torch


class AdjData(NamedTuple):
    """Stores the adjacency matrix and the node features of a graph."""

    adj_matrix: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor | float

    def __repr__(self) -> str:
        target = (
            self.y
            if isinstance(self.y, float)
            else self.y.size()
        )
        return (
            f"AdjData(adj_matrix={self.adj_matrix.size()}, "
            f"x={self.x.size()}, "
            f"y={target})"
        )
