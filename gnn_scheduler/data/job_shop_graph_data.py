from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Any

import torch
import torch_geometric as pyg
import networkx as nx

from gnn_scheduler.jssp import JobShopInstance
from gnn_scheduler.data.preprocessing import NodeFeatureCreator



class GraphDataConverter(ABC):
    """Base class for graph data converters.
    
    Converts a networkx's graph to a PyG graph data object."""

    def __init__(
        self,
        node_feature_creators: list[NodeFeatureCreator],
    ):
        self.node_feature_creators = node_feature_creators


class GraphTaskDataConverter(GraphDataConverter):
    """Converts a networkx's graph to a PyG graph data object."""

    def __init__(
        self,
        node_feature_creators: list[NodeFeatureCreator],
    ):
        super().__init__(node_feature_creators)

    def convert(self, graph: nx.DiGraph) -> pyg.data.Data:
        """Converts a networkx's graph to a PyG graph data object.

        Args:
            graph (nx.DiGraph): _description_

        Returns:
            pyg.data.Data: _description_
        """
        x = []
        for node_name, node_data in graph.nodes(data=True):
            node_features = []
            for node_feature_creator in self.node_feature_creators:
                node_features += node_feature_creator(
                    node_name, node_data, graph
                )
            x.append(node_features)
        x = torch.tensor(x, dtype=torch.float)

        edge_index = []
        for edge in graph.edges():
            edge_index.append(edge)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        return pyg.data.Data(x=x, edge_index=edge_index)
