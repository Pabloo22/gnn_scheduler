import os
from typing import Optional
import pickle

import tqdm

from gnn_scheduler import get_data_path
from gnn_scheduler.jssp import (
    JobShopInstance,
    load_pickle_instances_from_folders,
    load_pickle_objects_from_folders,
)
from gnn_scheduler.jssp.graphs import (
    NodeFeatureCreator,
    JobID,
    OperationIndex,
    OneHotEncoding,
    AdjData,
    disjunctive_graph_to_tensors,
    DisjunctiveGraph,
)
from gnn_scheduler.gnns.training import train_eval_test_split
from gnn_scheduler.difficulty_prediction import get_difficulty_score


def diff_pred_node_features_creators(n_machines: int = 10):
    """Retuns a OneHotEncoding node feature creator for machine_id,
    an OperationIndex, and JobID."""

    machine_one_hot = OneHotEncoding("machine_id", n_machines)
    operation_index = OperationIndex()
    job_id = JobID()
    return [machine_one_hot, operation_index, job_id]


def instance_to_adj_data(
    instance: JobShopInstance,
    node_feature_creators: list[NodeFeatureCreator],
    copy: bool = False,
    sparse: bool = True,
    directed: bool = False,
) -> AdjData:
    """Returns the node features and adjacency matrices of a job-shop instance.

    Args:
        instance (JobShopInstance): the instance.
        node_feature_creators (list[NodeFeatureCreator]): the node feature
            creators to use.
        copy (bool, optional): whether to copy the graph before preprocessing.
            Defaults to False.
        sparse (bool, optional): whether to use a sparse tensor for the
            adjacency matrix. Defaults to True.
        directed (bool, optional): whether the graph is directed. It only
            affects the conjunctive edges. Defaults to False.

    Returns:
        AdjData: the node features and adjacency matrices
    """
    y = get_difficulty_score(instance)
    disjunctive_graph = DisjunctiveGraph.from_job_shop_instance(instance)
    node_features, adj_matrices = disjunctive_graph_to_tensors(
        disjunctive_graph,
        node_feature_creators=node_feature_creators,
        copy=copy,
        sparse=sparse,
        directed=directed,
    )

    return AdjData(adj_matrix=adj_matrices, x=node_features, y=y)


def process_data(
    folder_names: list[str],
    show_progress: bool = True,
    data_path: Optional[os.PathLike | str | bytes] = None,
    sparse: bool = True,
    directed: bool = False,
) -> list[AdjData]:
    """Loads the data from the given folders as AdjData objects.

    Args:
        folder_names (list[str]): the names of the folders containing the
            instances
        show_progress (bool, optional): whether to show a progress bar.
            Defaults to True.
        data_path (Optional[os.PathLike | str | bytes], optional): the path to
            the data folder. Defaults to `gnn_scheduler.get_data_path()`.
        sparse (bool, optional): whether to use a sparse tensor for the
            adjacency matrix. Defaults to True.
        directed (bool, optional): whether the graph is directed. It only
            affects the conjunctive edges. Defaults to False.

    Returns:
        list[AdjData]: the AdjData objects
    """
    instances = load_pickle_instances_from_folders(
        folder_names, show_progress=show_progress, data_path=data_path
    )

    node_feature_creators = diff_pred_node_features_creators()

    adj_data_list = []
    for instance in tqdm.tqdm(
        instances, disable=not show_progress, desc="Creating AdjData objects"
    ):
        adj_data = instance_to_adj_data(
            instance, node_feature_creators, sparse=sparse, directed=directed
        )
        adj_data_list.append(adj_data)

    return adj_data_list


def save_adj_data_list(
    adj_data_list: list[AdjData],
    folder_name: str,
    show_progress: bool = True,
    data_path: Optional[os.PathLike | str | bytes] = None,
):
    """Saves a list of AdjData objects to a folder.

    Args:
        adj_data_list (list[AdjData]): the list of AdjData objects to save.
        folder_name (str): the name of the folder to save the AdjData objects
            to.
        show_progress (bool, optional): whether to show a progress bar.
            Defaults to True.
        data_path (Optional[os.PathLike | str | bytes], optional): the path to
            the data folder. Defaults to `gnn_scheduler.get_data_path()`.
    """
    # Create the folder if it doesn't exist
    if data_path is None:
        data_path = get_data_path()
    folder_path = os.path.join(data_path, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for i, adj_data in tqdm.tqdm(
        enumerate(adj_data_list),
        disable=not show_progress,
        desc="Saving AdjData objects",
    ):
        file_name = f"{i}.pkl"
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "wb") as f:
            pickle.dump(adj_data, f)


def load_and_split_data(
    folder_names: list[str],
    seed: int = 0,
    eval_size: float = 0.1,
    test_size: float = 0.2,
    show_progress: bool = True,
    data_path: Optional[os.PathLike | str | bytes] = None,
) -> tuple[list[AdjData], list[AdjData], list[AdjData]]:
    """Loads the data from the given folders, splits it into train, eval and
    test sets and returns them."""
    adj_data_list = load_pickle_objects_from_folders(
        folder_names, show_progress=show_progress, data_path=data_path
    )
    train_adj_data, eval_adj_data, test_adj_data = train_eval_test_split(
        adj_data_list,
        seed=seed,
        eval_size=eval_size,
        test_size=test_size,
    )
    return train_adj_data, eval_adj_data, test_adj_data
