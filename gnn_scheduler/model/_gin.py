from typing import Dict, List, Optional, Tuple
import itertools
import torch
from torch import nn
from torch_geometric.nn import GINConv, HeteroConv


class HeteroMetadata:

    __slots__ = ["node_types", "edge_types"]

    def __init__(
        self,
        node_types: List[str],
        edge_types: Optional[List[Tuple[str, str, str]]] = None,
    ):
        self.node_types = node_types

        if edge_types is None:
            edge_types = []
            for source, target in itertools.product(node_types, node_types):
                edge_types.append((source, "to", target))
        self.edge_types = edge_types


class HGINLayer(torch.nn.Module):
    """
    Heterogeneous Graph Isomorphism Network layer using PyG's built-in GINConv.
    """

    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        out_channels: int,
        metadata: HeteroMetadata,
        use_batch_norm: bool = True,
        aggregation: str = "sum",
    ):
        super().__init__()

        # Create MLPs for each node type
        self.mlps = nn.ModuleDict()
        for node_type in metadata.node_types:
            self.mlps[node_type] = nn.Sequential(
                nn.Linear(in_channels_dict[node_type], out_channels),
                (
                    nn.BatchNorm1d(out_channels)
                    if use_batch_norm
                    else nn.Identity()
                ),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
                (
                    nn.BatchNorm1d(out_channels)
                    if use_batch_norm
                    else nn.Identity()
                ),
                nn.ReLU(),
            )

        # Create heterogeneous convolution using GINConv for each edge type
        self.conv = HeteroConv(
            {
                edge_type: GINConv(self.mlps[edge_type[0]], train_eps=True)
                for edge_type in metadata.edge_types
            },
            aggr=aggregation,
        )

    def forward(self, x_dict, edge_index_dict):
        return self.conv(x_dict, edge_index_dict)


def initialize_hgin_layers(
    metadata: HeteroMetadata,
    in_channels_dict: Dict[str, int],
    hidden_channels: int,
    num_layers: int,
    use_batch_norm: bool,
) -> nn.ModuleList:
    convs = nn.ModuleList()
    for i in range(num_layers):
        conv_in_channels = (
            in_channels_dict
            if i == 0
            else {
                node_type: hidden_channels for node_type in metadata.node_types
            }
        )

        conv = HGINLayer(
            conv_in_channels,
            hidden_channels,
            metadata,
            use_batch_norm,
        )
        convs.append(conv)
    return convs
