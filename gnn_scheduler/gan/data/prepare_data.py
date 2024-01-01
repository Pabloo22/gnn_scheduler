import os
from typing import Optional
import pickle

import tqdm

from gnn_scheduler import get_data_path
from gnn_scheduler.jssp import (
    JobShopInstance,
    load_pickle_instances_from_folders,
)
from gnn_scheduler.jssp.graphs import (
    NodeFeatureCreator,
    JobID,
    OperationIndex,
    OneHotEncoding,
    disjunctive_graph_to_tensors,
    DisjunctiveGraph,
)
from gnn_scheduler.gan import get_difficulty_score
from gnn_scheduler.gan.data import DenseData


def diff_pred_node_features_creators(n_machines: int = 10):
    """Retuns a OneHotEncoding node feature creator for machine_id,
    an OperationIndex, and JobID."""

    machine_one_hot = OneHotEncoding("machine_id", n_machines)
    operation_index = OperationIndex()
    job_id = JobID()
    return [machine_one_hot, operation_index, job_id]


def instance_to_dense_data(
    instance: JobShopInstance,
    node_feature_creators: list[NodeFeatureCreator],
    copy: bool = False,
    sparse: bool = True,
    directed: bool = False,
) -> DenseData:
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
        DenseData: the node features and adjacency matrices
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

    return DenseData(adj_matrix=adj_matrices, x=node_features, y=y)


def process_data(
    folder_names: list[str],
    show_progress: bool = True,
    data_path: Optional[os.PathLike | str | bytes] = None,
    sparse: bool = True,
    directed: bool = False,
) -> list[DenseData]:
    """Loads the data from the given folders as DenseData objects.

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
        list[DenseData]: the DenseData objects
    """
    instances = load_pickle_instances_from_folders(
        folder_names, show_progress=show_progress, data_path=data_path
    )

    node_feature_creators = diff_pred_node_features_creators()

    dense_data_list = []
    for instance in tqdm.tqdm(
        instances, disable=not show_progress, desc="Creating DenseData objects"
    ):
        dense_data = instance_to_dense_data(
            instance, node_feature_creators, sparse=sparse, directed=directed
        )
        dense_data_list.append(dense_data)

    return dense_data_list


def process_and_save_data(
    folder_names: list[str],
    new_folder_names: list[str],
    show_progress: bool = True,
    data_path: Optional[os.PathLike | str | bytes] = None,
    sparse: bool = True,
    directed: bool = False,
    batch_size: int = 1000,
) -> list[DenseData]:
    """Loads the data from the given folders as DenseData objects and saves them
    to a new folder.

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
        batch_size (int, optional): the batch size. Defaults to 1000.

    Returns:
        list[DenseData]: the DenseData objects
    """
    instances = load_pickle_instances_from_folders(
        folder_names, show_progress=show_progress, data_path=data_path
    )

    node_feature_creators = diff_pred_node_features_creators()

    dense_data_list = []
    start_index = 0
    for instance in tqdm.tqdm(
        instances, disable=not show_progress, desc="Creating DenseData objects"
    ):
        dense_data = instance_to_dense_data(
            instance, node_feature_creators, sparse=sparse, directed=directed
        )
        dense_data_list.append(dense_data)

        if len(dense_data_list) % batch_size == 0:
            save_dense_data_list(
                dense_data_list,
                folder_name=new_folder_names[0],
                show_progress=True,
                data_path=data_path,
                start_index=start_index,
            )
            start_index += len(dense_data_list)
            dense_data_list = []

    return dense_data_list


def save_dense_data_list(
    dense_data_list: list[DenseData],
    folder_name: str,
    show_progress: bool = True,
    data_path: Optional[os.PathLike | str | bytes] = None,
    start_index: int = 0,
):
    """Saves a list of DenseData objects to a folder.

    Args:
        dense_data_list (list[DenseData]): the list of DenseData
            objects to save.
        folder_name (str): the name of the folder to save the
            DenseData objects to.
        show_progress (bool, optional): whether to show a progress bar.
            Defaults to True.
        data_path (Optional[os.PathLike | str | bytes], optional): the path to
            the data folder. Defaults to `gnn_scheduler.get_data_path()`.
        start_index (int, optional): the index to start from. Defaults to 0.
    """
    # Create the folder if it doesn't exist
    if data_path is None:
        data_path = get_data_path()
    folder_path = os.path.join(data_path, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for i, dense_data in tqdm.tqdm(
        enumerate(dense_data_list),
        disable=not show_progress,
        desc="Saving DenseData objects",
    ):
        file_name = f"{start_index + i}.pkl"
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "wb") as f:
            pickle.dump(dense_data, f)
