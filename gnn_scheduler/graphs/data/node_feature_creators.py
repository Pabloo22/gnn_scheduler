from __future__ import annotations

# from abc import ABC, abstractmethod
from typing import Any, Callable

import networkx as nx


NodeFeatureCreator = Callable[[str, dict[str, Any], nx.DiGraph], list[float]]

# class NodeFeatureCreator(ABC):
#     """Base class for node feature creators.
#     Args:
#         ABC (_type_): _description_
#     """

#     @abstractmethod
#     def create_features(
#         self, node_name: str, node_data: dict[str, Any], graph: nx.DiGraph
#     ) -> list[float]:
#         """_summary_

#         Args:
#             node_name (str): _description_
#             node_data (dict[str, Any]): _description_
#             graph (nx.DiGraph): _description_

#         Returns:
#             list[float]: _description_
#         """

#     def __call__(
#         self, node_name: str, node_data: dict[str, Any], graph: nx.DiGraph
#     ) -> list[float]:
#         return self.create_features(node_name, node_data, graph)



def normalized_in_out_degrees(
    self, node_name: str, node_data: dict[str, Any], graph: nx.DiGraph
) -> list[float]:
    """Returns the normalized in- and out-degrees of a node.

    Args:
        node_name (str): _description_
        node_data (dict[str, Any]): _description_
        graph (nx.DiGraph): _description_

    Returns:
        list[float]: _description_
    """
    in_degree = graph.in_degree(node_name) / (graph.number_of_nodes() - 1)
    out_degree = graph.out_degree(node_name) / (graph.number_of_nodes() - 1)
    return [in_degree, out_degree]


class OneHotEncoding(NodeFeatureCreator):
    """One-hot encoding of a node attribute.

    Args:
        NodeFeatureCreator (_type_): _description_
    """

    def __init__(self, feature_name: str, n_values: int = 100):
        self.feature_name = feature_name
        self.n_values = n_values

    def __call__(
        self, node_name: str, node_data: dict[str, Any], graph: nx.DiGraph
    ) -> list[float]:
        """_summary_

        Args:
            node_name (str): _description_
            node_data (dict[str, Any]): _description_
            graph (nx.DiGraph): _description_

        Returns:
            list[float]: _description_
        """
        zeros = [0.0] * self.n_values
        machine_id = node_data.get(self.feature_name)
        if machine_id is not None and 0 <= machine_id < self.n_values:
            zeros[machine_id] = 1.0
        return zeros
