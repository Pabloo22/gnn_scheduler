"""Layers that are designed to handle graph data with multiple edge types
and adjacency matrices, instead of edge_index lists.

Code adapted from:
https://github.com/harutatsuakiyama/Implementation-MolGAN-PyTorch/blob/master/layers.py

under the CC-BY-4.0 license:
https://choosealicense.com/licenses/cc-by-4.0/

Comments and the GraphAggregationLayer are my own.
"""
from __future__ import annotations

from typing import Optional, Callable

import torch
import torch.nn as nn


class GraphConvolutionLayer(nn.Module):
    """graph convolution layer capable of processing graph data with
    multiple edge types.

    It performs feature transformations for each edge type, aggregates these
    features based on the graph structure, and then combines these aggregated
    features with directly transformed node features. This approach allows
    the model to learn complex representations of nodes in a graph,
    considering the different types of relationships between them.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Callable,
        edge_type_num: int = 2,
        dropout_rate: float = 0.0,
    ):
        """Initialize the GraphConvolutionLayer class.

        Args:
            in_features: The number of features for each node in the input graph.
            out_features: The number of output features for each node.
            activation: The activation function to be applied (e.g., ReLU, sigmoid).
            edge_type_num: The number of different types of edges in the graph.
            dropout_rate: The dropout rate for regularization.
        """
        super().__init__()
        self.edge_type_num = edge_type_num
        self.out_features = out_features
        self.adj_list = nn.ModuleList()
        for _ in range(self.edge_type_num):
            self.adj_list.append(nn.Linear(in_features, out_features))
        self.linear_2 = nn.Linear(in_features, out_features)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        n_tensor: torch.Tensor,
        adj_tensor: torch.Tensor,
        h_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Defines the computation performed at every call.

        Args:
            n_tensor (Tensor): The node features tensor.
            adj_tensor (Tensor): The adjacency tensor representing the
                connections in the graph.
            h_tensor (Tensor): An optional tensor of additional node features.

        Returns:
            Tensor: The output node features tensor.
        """
        if h_tensor is not None:
            node_features = torch.cat((n_tensor, h_tensor), -1)
        else:
            node_features = n_tensor
        # output is computed by applying each linear transformation in
        # self.adj_list to node_features. This results in a tensor with a
        # separate set of transformed features for each edge type
        output = torch.stack(
            [
                self.adj_list[i](node_features)
                for i in range(self.edge_type_num)
            ],
            1,
        )
        # These transformed features are then multiplied by the adjacency
        # tensor adj_tensor to aggregate features across the graph,
        # considering different edge types.
        output = torch.matmul(adj_tensor, output)

        # out_sum is the sum of these aggregated features.
        out_sum = torch.sum(output, 1)

        out_linear_2 = self.linear_2(node_features)

        output = out_sum + out_linear_2
        output = (
            self.activation(output) if self.activation is not None else output
        )
        output = self.dropout(output)
        return output


class MultiGraphConvolutionLayers(nn.Module):
    """Creates a sequence of graph convolutional layers, each capable
    of handling multiple types of edges in a graph.

    This module can be used to build deep graph neural networks by stacking
    multiple graph convolution layers.

    Attributes:
        conv_nets (nn.ModuleList): A list of graph convolution layers.
        units (List[int]): Number of output units for each graph convolution
            layer.

    Args:
        in_features (int): Number of features for each node in the input graph.
        units (List[int]): Number of output units for each layer.
        activation (Callable): Activation function to be applied after each
            layer.
        edge_type_num (int, optional): Number of different types of edges in
            the graph. Defaults to 2.
        with_features (bool, optional): If True, additional features are
            considered. Defaults to False.
        feature_dim_size (int, optional): Additional feature dimension size.
            Defaults to 0.
        dropout_rate (float, optional): Dropout rate for regularization.
            Defaults to 0.
    """

    def __init__(
        self,
        in_features: int,
        units: list[int],
        activation: callable,
        edge_type_num: int = 2,
        with_features: bool = False,
        feature_dim_size: int = 0,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.conv_nets = nn.ModuleList()
        self.units = units
        in_units = []

        # Adjust input units based on whether additional features are used
        if with_features:
            in_units = [x + in_features for x in self.units]
            input_sizes = [in_features + feature_dim_size] + in_units[:-1]
        else:
            in_units = [x + in_features for x in self.units]
            input_sizes = [in_features] + in_units[:-1]

        # Create graph convolution layers
        for u0, u1 in zip(input_sizes, self.units):
            self.conv_nets.append(
                GraphConvolutionLayer(
                    u0, u1, activation, edge_type_num, dropout_rate
                )
            )

    def forward(
        self,
        n_tensor: torch.Tensor,
        adj_tensor: torch.Tensor,
        h_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the multi-layer graph convolution network.

        Args:
            n_tensor (torch.Tensor): The node features tensor.
            adj_tensor (torch.Tensor): The adjacency tensor representing the
                connections in the graph.
            h_tensor (torch.Tensor, optional): An optional tensor of
                additional node features.

        Returns:
            torch.Tensor: Output tensor after passing through the graph
                convolution layers.
        """
        hidden_tensor = h_tensor
        for conv_layer in self.conv_nets:
            hidden_tensor = conv_layer(n_tensor, adj_tensor, hidden_tensor)

        return hidden_tensor


class GraphConvolution(nn.Module):
    """Serves as an interface for utilizing multiple  graph convolution layers
    together.

    This class wraps the MultiGraphConvolutionLayers class and allows 
    the creation of a graph convolution network that can handle multiple
    types of edges and additional features.

    Attributes:
        in_features (int): Number of input features per node.
        graph_conv_units (List[int]): Number of units in each graph
            convolution layer.
        activation_f (torch.nn.Module): Activation function applied
            in each layer.
        multi_graph_convolution_layers (MultiGraphConvolutionLayers): The
            multi-layer graph convolution component.

    Args:
        in_features (int): Number of features for each node in the input graph.
        graph_conv_units (List[int]): A list specifying the number of units
            in each graph convolution layer.
        edge_type_num (int): Number of different types of edges in the graph.
                with_features (bool, optional): If True, additional features are considered.
            Defaults to False.
        f_dim (int, optional): Size of additional feature dimensions. 
            Defaults to 0.
        dropout_rate (float, optional): Dropout rate for regularization. 
            Defaults to 0.
        dropout_rate (float, optional): Dropout rate for regularization. 
            Defaults to 0.
    """

    def __init__(
        self,
        in_features: int,
        graph_conv_units: [int],
        edge_type_num: int,
        with_features: bool = False,
        f_dim: int = 0,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.graph_conv_units = graph_conv_units
        self.activation_f = torch.nn.Tanh()
        self.multi_graph_convolution_layers = MultiGraphConvolutionLayers(
            in_features,
            self.graph_conv_units,
            self.activation_f,
            edge_type_num,
            with_features,
            f_dim,
            dropout_rate,
        )

    def forward(
        self,
        n_tensor: torch.Tensor,
        adj_tensor: torch.Tensor,
        h_tensor: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass through the Graph Convolution network.

        Args:
            n_tensor (torch.Tensor): The node features tensor.
            adj_tensor (torch.Tensor): The adjacency tensor representing
                connections in the graph.
            h_tensor (torch.Tensor, optional): An optional tensor of
                additional node features.

        Returns:
            torch.Tensor: Output tensor after passing through the graph
                convolution network.
        """
        output = self.multi_graph_convolution_layers(
            n_tensor, adj_tensor, h_tensor
        )
        return output


class GraphAggregationLayer(nn.Module):
    """Combines node embeddings using sum, mean, and global max pooling, and
    then processes the combined embedding through an MLP.

    Attributes:
        mlp (nn.Module): A Multi-Layer Perceptron for processing the aggregated embeddings.

    Args:
        in_features (int): The number of features in each input node embedding.
        out_features (int): The number of features in the output node embedding.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        leaky_relu_slope: float = 0.1,
    ):
        super().__init__()
        # Embedding sizes for sum, mean, and max pooling
        embedding_size = in_features * 3

        # MLP with a single output neuron
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, out_features),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(out_features, 1),
        )

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the graph aggregation layer.

        Args:
            node_features (torch.Tensor): The node features tensor of shape
                (N, in_features), where N is the number of nodes.

        Returns:
            torch.Tensor: The output tensor of shape (1, 1)
        """

        # Sum, Mean, and Max Pooling
        sum_embedding = torch.sum(node_features, dim=0, keepdim=True)
        mean_embedding = torch.mean(node_features, dim=0, keepdim=True)
        max_embedding, _ = torch.max(node_features, dim=0, keepdim=True)

        # Concatenate the embeddings
        combined_embedding = torch.cat(
            (sum_embedding, mean_embedding, max_embedding), dim=1
        )

        # Process the combined embedding through the MLP
        output = self.mlp(combined_embedding)

        return output
