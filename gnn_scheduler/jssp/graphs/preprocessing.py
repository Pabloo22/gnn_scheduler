from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np
import torch

from gnn_scheduler.jssp.graphs import (
    NodeFeatureCreator,
    DisjunctiveGraph,
    EdgeType,
)


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


def get_adj_matrices(
    graph: DisjunctiveGraph, directed: bool = True
) -> torch.Tensor:
    """Returns a tensor of adjacency matrices of a disjunctive graph.

    The first adjacency matrix is the adjacency matrix of the conjuctive edges
    and the second adjacency matrix is  of the disjunctive edges.

    The tensor shape is (2, num_nodes, num_nodes).
    Args:
        graph (DisjunctiveGraph): the disjunctive graph
        directed (bool, optional): whether the graph is directed. It only
            affects the conjunctive edges. Defaults to True.

    Returns:
        torch.Tensor: the tensor of adjacency matrices
    """
    num_nodes = graph.number_of_nodes()
    adj_matrices = np.zeros((2, num_nodes, num_nodes))
    for u, v, edge_type in graph.edges(data="type"):
        type_index = edge_type.value
        # get data from u and v
        u_index = graph.nodes[u]["node_index"]
        v_index = graph.nodes[v]["node_index"]
        adj_matrices[type_index, u_index, v_index] = 1

        if not directed and type_index == EdgeType.CONJUNCTIVE.value:
            adj_matrices[type_index, v_index, u_index] = 1

    return torch.tensor(adj_matrices)


def get_sparse_adj_matrices(
    graph: DisjunctiveGraph, directed: bool = False
) -> torch.Tensor:
    """Returns a tensor of adjacency matrices of a disjunctive graph.

    The first adjacency matrix is the adjacency matrix of the conjunctive edges,
    and the second adjacency matrix is of the disjunctive edges.

    The tensor shape is (2, num_nodes, num_nodes), and the result is returned as sparse tensors.
    Args:
        graph (DisjunctiveGraph): the disjunctive graph
        directed (bool, optional): whether the graph is directed. It only
            affects the conjunctive edges. Defaults to False.

    Returns:
        torch.Tensor: the tensor of adjacency matrices
    """
    num_nodes = graph.number_of_nodes()

    # Initialize lists to store indices and values for each type of edge
    indices_conj = []
    indices_disj = []

    for u, v, edge_type in graph.edges(data="type"):
        u_index = graph.nodes[u]["node_index"]
        v_index = graph.nodes[v]["node_index"]

        # Store indices based on edge type
        if edge_type.value == EdgeType.CONJUNCTIVE.value:
            indices_conj.append([u_index, v_index])
            if not directed:
                indices_conj.append([v_index, u_index])
        else:
            indices_disj.append([u_index, v_index])

    # Convert lists to tensors
    indices_conj = (
        torch.tensor(indices_conj).t().contiguous()
        if indices_conj
        else torch.empty((2, 0))
    )
    indices_disj = (
        torch.tensor(indices_disj).t().contiguous()
        if indices_disj
        else torch.empty((2, 0))
    )
    values_conj = torch.ones(indices_conj.size(1))
    values_disj = torch.ones(indices_disj.size(1))

    # Create sparse tensors
    adj_matrix_conj = torch.sparse_coo_tensor(
        indices_conj, values_conj, (num_nodes, num_nodes)
    )
    adj_matrix_disj = torch.sparse_coo_tensor(
        indices_disj, values_disj, (num_nodes, num_nodes)
    )

    # Stack the sparse tensors
    adj_matrices = torch.stack([adj_matrix_conj, adj_matrix_disj])

    return adj_matrices


def disjunctive_graph_to_tensors(
    disjunctive_graph: DisjunctiveGraph,
    node_feature_creators: list[NodeFeatureCreator],
    copy: bool = False,
    sparse: bool = False,
    directed: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns the node features and adjacency matrices of a job-shop instance.

    Args:
        disjunctive_graph (DisjunctiveGraph): the disjunctive graph of the
            instance.
        node_feature_creators (list[NodeFeatureCreator]): the node feature
            creators to use.
        copy (bool, optional): whether to copy the graph before preprocessing.
            Defaults to False.
        sparse (bool, optional): whether to return sparse tensors. Defaults to
            False.
        directed (bool, optional): whether the graph is directed. It only
            affects the conjunctive edges. Defaults to False.

    Returns:
        tuple[torch.Tensor]: the node features and adjacency matrices
    """

    disjunctive_graph = preprocess_graph(
        disjunctive_graph,
        node_feature_creators=node_feature_creators,
        new_feature_name="x",
        keep_old_features=False,
        exclude_old_features=["node_index"],
        copy=copy,
        remove_nodes=["T", "S"],
    )

    node_features = get_node_features_matrix(disjunctive_graph)

    if not sparse:
        adj_matrices = get_adj_matrices(disjunctive_graph)
    else:
        adj_matrices = get_sparse_adj_matrices(
            disjunctive_graph, directed=directed
        )

    return node_features, adj_matrices
