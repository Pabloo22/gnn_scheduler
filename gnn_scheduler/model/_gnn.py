from typing import Dict, List, Optional

import torch
from torch import nn
from torch_geometric.data import HeteroData

from gnn_scheduler.model import initialize_hgin_layers, HeteroMetadata

from job_shop_lib.graphs import NodeType


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
        in_channels_dict: Dict[str, int],
        hidden_channels: int = 256,
        num_layers: int = 3,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.convs = initialize_hgin_layers(
            metadata,
            in_channels_dict,
            hidden_channels,
            num_layers,
            use_batch_norm,
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
        self, data: HeteroData, valid_pairs: torch.Tensor
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
        # Initial node features
        x_dict = {
            node_type: data[node_type].x for node_type in data.node_types
        }

        # Store residual connections
        residuals: List[Dict[str, torch.Tensor]] = []

        # Graph convolutions
        for conv in self.convs:
            x_dict_new: Dict[str, torch.Tensor] = conv(
                x_dict, data.edge_index_dict
            )

            # Add residual connection
            if residuals:
                for node_type in x_dict_new:
                    x_dict_new[node_type] = (
                        x_dict_new[node_type] + residuals[-1][node_type]
                    )

            residuals.append(x_dict_new)
            x_dict = x_dict_new

        # Select valid pairs
        mapping = {
            NodeType.OPERATION.value: 0,
            NodeType.MACHINE.value: 1,
            NodeType.JOB.value: 2,
        }
        scores = torch.zeros(len(valid_pairs), device=data.x.device)
        concat_features_list = []
        for node_type in data.node_types:
            indices = valid_pairs[:, mapping[node_type]]
            concat_features_list.append(x_dict[node_type][indices])

        concat_features = torch.cat(concat_features_list, dim=1)
        scores = self.score_mlp(concat_features)

        return scores
