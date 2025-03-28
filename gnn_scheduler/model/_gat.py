from typing import Dict
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, HeteroConv
from torch_geometric.utils import dropout_edge

from gnn_scheduler.model import HeteroMetadata


class HGATV2Layer(torch.nn.Module):
    """Heterogeneous Graph Convolutional Network layer using PyG's built-in
    GATv2Conv.

    The ``__call__`` method of this module expects a dictionary of node
    features and a dictionary of edge indices.
    """

    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        out_channels: int,
        metadata: HeteroMetadata,
        use_batch_norm: bool = True,
        aggregation: str = "max",
        edge_dropout: float = 0.0,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        residual: bool = True,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.edge_dropout = edge_dropout

        # Create GATv2 convolutions for each edge type
        self.conv = HeteroConv(
            {
                edge_type: GATv2Conv(
                    in_channels=in_channels_dict[edge_type[0]],
                    out_channels=out_channels,
                    heads=heads,
                    concat=concat,
                    dropout=dropout,
                    residual=residual,
                    negative_slope=negative_slope,
                    add_self_loops=edge_type[0] == edge_type[2],
                )
                for edge_type in metadata.edge_types
            },
            aggr=aggregation,
        )

        # Create batch normalization layers for each node type
        self.batch_norms = nn.ModuleDict()
        if use_batch_norm:
            for node_type in metadata.node_types:
                self.batch_norms[node_type] = nn.BatchNorm1d(out_channels)

        # ReLU activation
        self.elu = nn.ELU()

    def forward(self, x_dict, edge_index_dict):
        # Apply edge dropout during training
        if self.training and self.edge_dropout > 0:
            # Create a new edge_index_dict with dropped edges
            dropped_edge_index_dict = {}

            for edge_type, edge_index in edge_index_dict.items():
                # Apply dropout_edge to each edge type
                dropped_edge_index, _ = dropout_edge(
                    edge_index,
                    p=self.edge_dropout,
                    force_undirected=False,
                    training=self.training,
                )
                dropped_edge_index_dict[edge_type] = dropped_edge_index

            # Use the dropped edges
            edge_index_dict = dropped_edge_index_dict

        # Apply convolution
        out_dict = self.conv(x_dict, edge_index_dict)

        # Apply batch normalization and ReLU for each node type
        if hasattr(self, "batch_norms"):
            for node_type in out_dict.keys():
                if node_type in self.batch_norms:
                    out_dict[node_type] = self.batch_norms[node_type](
                        out_dict[node_type]
                    )

        # Apply ReLU activation
        for node_type in out_dict.keys():
            out_dict[node_type] = self.elu(out_dict[node_type])

        return out_dict


def initialize_hgatv2_layers(
    metadata: HeteroMetadata,
    in_channels_dict: Dict[str, int],
    hidden_channels: int,
    num_layers: int,
    use_batch_norm: bool,
    aggregation: str = "max",
    edge_dropout: float = 0.0,
    heads: int = 1,
    concat: bool = True,
    dropout: float = 0.0,
    residual: bool = True,
    negative_slope: float = 0.2,
) -> nn.ModuleList:
    """Returns a ``ModuleList`` of ``HGCNLayer`` instances."""
    convs = nn.ModuleList()
    for i in range(num_layers):
        conv_in_channels = (
            in_channels_dict
            if i == 0
            else {
                node_type: hidden_channels for node_type in metadata.node_types
            }
        )

        conv = HGATV2Layer(
            in_channels_dict=conv_in_channels,
            out_channels=hidden_channels,
            metadata=metadata,
            use_batch_norm=use_batch_norm,
            aggregation=aggregation,
            edge_dropout=edge_dropout,
            heads=heads,
            concat=concat,
            dropout=dropout,
            residual=residual,
            negative_slope=negative_slope,
        )
        convs.append(conv)
    return convs
