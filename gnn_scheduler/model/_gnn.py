import torch
from torch import nn
from torch_geometric.data import HeteroData

from gnn_scheduler.model import (
    initialize_hgin_layers,
    HeteroMetadata,
    FeatureType,
    MultiPeriodicEncoder,
)


class ResidualSchedulingGNN(nn.Module):
    """
    Main GNN model for Residual Scheduling using PyG's built-in GINConv.

    Args:
        metadata:
            Graph metadata (node types and edge types)
        in_channels_dict:
            Dictionary of input feature dimensions for each node type
        hidden_channels:
            Hidden dimension size
        num_layers:
            Number of HGIN layers
        use_batch_norm:
            Whether to use batch normalization in MLPs
    """

    def __init__(
        self,
        metadata: HeteroMetadata,
        in_channels_dict: dict[str, int],
        initial_node_features_dim: int = 32,
        sigma: float = 1.0,
        hidden_channels: int = 64,
        num_layers: int = 3,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.metadata = metadata
        self.encoders = nn.ModuleDict(
            {
                node_type: MultiPeriodicEncoder(
                    in_channels_dict[node_type],
                    initial_node_features_dim,
                    concat=True,
                    sigma=sigma,
                )
                for node_type in metadata.node_types
            }
        )

        self.convs = initialize_hgin_layers(
            metadata,
            in_channels_dict={
                node_type: initial_node_features_dim
                for node_type in metadata.node_types
            },
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            use_batch_norm=use_batch_norm,
        )

        # Score function MLP
        score_in_channels = hidden_channels * len(metadata.node_types)
        self.score_mlp = nn.Sequential(
            nn.Linear(score_in_channels, hidden_channels),
            (
                nn.BatchNorm1d(hidden_channels)
                if use_batch_norm
                else nn.Identity()
            ),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels // 2),
            (
                nn.BatchNorm1d(hidden_channels // 2)
                if use_batch_norm
                else nn.Identity()
            ),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
        valid_pairs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            data: Heterogeneous graph data
            valid_pairs:
                Tensor of valid (operation_id, machine_id, job_id) pairs to
                score. If ``None``, scores all possible pairs.
                Shape: [num_pairs, 3]

        Returns:
            Tensor of scores for operation-machine-job pairs
        """

        # Store residual connections
        residuals: list[dict[str, torch.Tensor]] = []

        x_dict = {
            node_type: self.encoders[node_type](x)
            for node_type, x in x_dict.items()
        }
        # Graph convolutions
        node_type = None
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

            # Add residual connection
            if residuals:
                for node_type in x_dict:
                    x_dict[node_type] = (
                        x_dict[node_type] + residuals[-1][node_type]
                    )

            residuals.append(x_dict)

        # Select valid pairs
        mapping = {
            "operation": 0,
            "machine": 1,
            "job": 2,
        }
        assert node_type is not None
        scores = torch.zeros(len(valid_pairs), device=x_dict[node_type].device)
        concat_features_list = []
        for node_type in self.metadata.node_types:
            indices = valid_pairs[:, mapping[node_type]]
            concat_features_list.append(x_dict[node_type][indices])

        concat_features = torch.cat(concat_features_list, dim=1)
        scores = self.score_mlp(concat_features)

        return scores
