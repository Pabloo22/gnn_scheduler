"""Contains models that use"""
from typing import Optional

import torch
from torch import nn

from gnn_scheduler.gnns.models.dense_matrix import (
    GraphAggregation,
    MultiGraphConvolutionLayers,
    GraphConvolution,
    MyGraphAggregationLayer,
    MultiDenseLayer,
)
from gnn_scheduler.gnns import get_activation_function


class RelationalGCNRegressor(nn.Module):
    """
    A regression model that processes graphs with two types of edges using a
    relational graph convolutional network (GCN).

    Attributes:
        graph_conv (MultiGraphConvolutionLayers): The graph convolution layers.
        aggregation_layer (GraphAggregationLayer): The layer that aggregates node embeddings.
    """

    def __init__(
        self,
        in_features: int,
        conv_units: list[int],
        aggregation_units: int,
        dropout_rate: float = 0.0,
        with_features: bool = False,
        feature_dim_size: int = 0,
    ):
        """Initializes the RelationalGCNRegressor model.

        Args:
            in_features (int): The number of features in each input node
                embedding.
            conv_units ([int]): List of units in each graph convolution layer.
            aggregation_units (int): The number of units in the graph
                aggregation layer before regression.
            dropout_rate (float, optional): Dropout rate for the convolution
                layers. Defaults to 0.
            with_features (bool, optional): Whether to use additional
                node features. Defaults to False.
            feature_dim_size (int, optional): The number of units in the
                additional node features. Defaults to 0.
        """
        super().__init__()

        # Define the graph convolution layers
        self.graph_conv = MultiGraphConvolutionLayers(
            in_features,
            conv_units,
            torch.nn.Tanh(),
            edge_type_num=2,
            dropout_rate=dropout_rate,
            with_features=with_features,
            feature_dim_size=feature_dim_size,
        )

        # Define the aggregation layer
        # The input size is the output size of the last graph convolution layer
        self.aggregation_layer = MyGraphAggregationLayer(
            conv_units[-1],
            aggregation_units,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        adj_matrices: torch.Tensor,
        h_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the RelationalGCNRegressor.

        Args:
            node_features (torch.Tensor): The node features tensor of shape
                (N, in_features), where N is the number of nodes.
            adj_matrices ([torch.Tensor]): The adjacency matrices of the graph.
        Returns:
            torch.Tensor: The regression output for the graph.
        """
        # Graph convolution layers
        hidden_features = self.graph_conv(
            node_features, adj_matrices, h_tensor
        )

        # Aggregation layer
        output = self.aggregation_layer(hidden_features)

        # Use sigmoid to get a value between 0 and 1
        output = torch.sigmoid(output)

        return output


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(
        self,
        conv_dim,
        m_dim,
        b_dim,
        with_features=False,
        f_dim=0,
        dropout_rate=0.0,
        out_activation_f: Optional[nn.Module | str] = None,
    ):
        super().__init__()
        if isinstance(out_activation_f, str):
            out_activation_f = get_activation_function(out_activation_f)
        self.out_activation_f = out_activation_f
        self.activation_f = torch.nn.Tanh()
        graph_conv_dim, aux_dim, linear_dim = conv_dim
        # discriminator
        self.gcn_layer = GraphConvolution(
            m_dim, graph_conv_dim, b_dim, with_features, f_dim, dropout_rate
        )
        self.agg_layer = GraphAggregation(
            graph_conv_dim[-1] + m_dim,
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
