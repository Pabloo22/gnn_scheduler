from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np
import torch

from gnn_scheduler.jssp.graphs import NodeFeatureCreator, EdgeType, DisjunctiveGraph


def preprocess_graph(
    graph: nx.DiGraph,
    node_feature_creators: list[NodeFeatureCreator],
    new_feature_name: str = "x",
    keep_old_features: bool = False,
    exclude_old_features: Optional[list[str]] = None,
    copy: bool = False,
    remove_nodes: Optional[list[str]] = None,
) -> nx.DiGraph:
    """Preprocesses a graph using a list of node feature creators.

    It creates a new feature for each node using the node feature creators.
    The new feature is stored in the node data under the name new_feature_name.
    If keep_old_features is True, the old features are kept. Otherwise, they
    are removed. If exclude_old_features is not None, the old features to
    exclude are specified.

    Args:
        graph (nx.DiGraph): the graph to preprocess
        node_feature_creators (list[NodeFeatureCreator]): the node feature
            creators to use.
        new_feature_name (str, optional): the name of the new feature. Defaults
            to "x".
        keep_old_features (bool, optional): whether to keep the old features.
            Defaults to False.
        exclude_old_features (Optional[list[str]], optional): the old features
            to exclude if keep_old_features is False. Defaults to None.
        copy (bool, optional): whether to copy the graph before preprocessing.
            Defaults to False.
        remove_nodes (Optional[list[str]], optional): the nodes names to
            remove. Defaults to [].

    Returns:
        nx.DiGraph: the preprocessed graph
    """

    if copy:
        graph = graph.copy()

    remove_nodes = [] if remove_nodes is None else remove_nodes

    for node_name in remove_nodes:
        graph.remove_node(node_name)

    exclude_old_features = (
        [] if exclude_old_features is None else exclude_old_features
    )
    exclude_old_features = set(exclude_old_features)
    exclude_old_features.add(new_feature_name)

    # Fit the node feature creators
    for node_feature_creator in node_feature_creators:
        node_feature_creator.fit(graph)

    for node_name, node_data in graph.nodes(data=True):
        # Add the new feature
        features = []
        for node_feature_creator in node_feature_creators:
            features.extend(node_feature_creator(node_name, node_data))
        node_data[new_feature_name] = features

        # Remove old features if necessary
        if keep_old_features:
            continue
        for feature_name in node_data.copy():
            if feature_name in exclude_old_features:
                continue
            del node_data[feature_name]

    return graph


def preprocess_graphs(
    graphs: list[nx.DiGraph],
    node_feature_creators: list[NodeFeatureCreator],
    new_feature_name: str = "x",
    keep_old_features: bool = False,
    exclude_old_features: Optional[list[str]] = None,
    copy: bool = False,
    remove_nodes: Optional[list[str]] = None,
) -> list[nx.DiGraph]:
    """Preprocesses a list of graphs using a list of node feature creators.

    Args:
        graphs (list[nx.DiGraph]): the graphs to preprocess
        node_feature_creators (list[NodeFeatureCreator]): the node feature
            creators to use.
        new_feature_name (str, optional): the name of the new feature. Defaults
            to "x".
        keep_old_features (bool, optional): whether to keep the old features.
            Defaults to False.
        exclude_old_features (Optional[list[str]], optional): the old features
            to exclude if keep_old_features is False. Defaults to None.
        copy (bool, optional): whether to copy the graph before preprocessing.
            Defaults to False.
        remove_nodes (Optional[list[str]], optional): the nodes names to
            remove. Defaults to [].

    Returns:
        list[nx.DiGraph]: the preprocessed graphs
    """
    processed_graphs = []
    for graph in graphs:
        processed_graphs.append(
            preprocess_graph(
                graph,
                node_feature_creators=node_feature_creators,
                new_feature_name=new_feature_name,
                keep_old_features=keep_old_features,
                exclude_old_features=exclude_old_features,
                copy=copy,
                remove_nodes=remove_nodes,
            )
        )
    return processed_graphs


def get_node_features_matrix(
    graph: nx.DiGraph, feature_name: str = "x"
) -> torch.Tensor:
    """Returns a tensor with the node features matrix of a graph.

    Args:
        graph (nx.DiGraph): the graph
        feature_name (str, optional): the name of the feature. Defaults to "x".

    Returns:
        
    """
    node_features_matrix = []
    for _, node_data in graph.nodes(data=True):
        node_features_matrix.append(node_data[feature_name])

    return torch.tensor(node_features_matrix)


def get_adj_matrices(graph: DisjunctiveGraph) -> torch.Tensor:
    """Returns a tensor of adjacency matrices of a disjunctive graph.
    
    The first adjacency matrix is the adjacency matrix of the conjuctive edges
    and the second adjacency matrix is  of the disjunctive edges.
    
    The tensor shape is (num_nodes, num_nodes, 2).
    Args:
        graph (DisjunctiveGraph): the disjunctive graph
    
    Returns:
        torch.Tensor: the tensor of adjacency matrices
    """
    num_nodes = graph.number_of_nodes()
    adj_matrices = np.zeros((num_nodes, num_nodes, 2))
    for u, v, edge_type in graph.edges(data="type"):
        type_index = edge_type.value
        # get data from u and v
        u_index = graph.nodes[u]["node_index"]
        v_index = graph.nodes[v]["node_index"]
        adj_matrices[u_index, v_index, type_index] = 1

    return torch.tensor(adj_matrices, dtype=torch.int8)
