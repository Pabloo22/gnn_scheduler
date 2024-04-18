"""Contains models that use"""
from typing import Optional

import torch
from torch import nn

from gnn_scheduler.gan.model import (
    GraphConvolution,
    GraphAggregation,
    MultiDenseLayer,
)
from gnn_scheduler.training_utils import get_activation_function


class Discriminator(nn.Module):
    """Discriminator/Reward network."""

    def __init__(
        self,
        graph_conv_dim: list[int],
        aux_dim: int,
        linear_dim: list[int],
        n_node_features: int,
        n_edge_types: int,
        with_features: bool = False,
        f_dim: int = 0,
        dropout_rate: float = 0.0,
        out_activation_f: Optional[nn.Module | str] = None,
    ):
        super().__init__()
        if isinstance(out_activation_f, str):
            out_activation_f = get_activation_function(out_activation_f)
        self.out_activation_f = out_activation_f
        self.activation_f = torch.nn.Tanh()

        self.gcn_layer = GraphConvolution(
            n_node_features, graph_conv_dim, n_edge_types, with_features, f_dim, dropout_rate
        )
        self.agg_layer = GraphAggregation(
            graph_conv_dim[-1] + n_node_features,
            aux_dim,
            self.activation_f,
            with_features,
            f_dim,
            dropout_rate,
        )
        self.multi_dense_layer = MultiDenseLayer(
            aux_dim, linear_dim, self.activation_f, dropout_rate=dropout_rate
        )

        self.output_layer = nn.Linear(linear_dim[-1], 1)

    def forward(self, adj_matrices, node_features, hidden=None):
        # adj = adj[:, :, :, 1:].permute(0, 3, 1, 2)
        h_1 = self.gcn_layer(node_features, adj_matrices, hidden)
        h = self.agg_layer(h_1, node_features, hidden)
        h = self.multi_dense_layer(h)

        output = self.output_layer(h)
        if self.out_activation_f is not None:
            output = self.out_activation_f(output)

        return output
