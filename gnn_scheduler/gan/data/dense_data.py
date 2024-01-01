from typing import NamedTuple, Optional

import torch


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
