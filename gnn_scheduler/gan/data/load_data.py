import os
from typing import Optional

import torch

from gnn_scheduler import get_project_path
from gnn_scheduler.job_shop import (
    load_pickle_object,
    load_pickle_objects_from_folders,
)
from gnn_scheduler.training_utils import train_eval_test_split
from gnn_scheduler.gan.data import DenseData


def remove_one_hot(node_features: torch.Tensor) -> torch.Tensor:
    """Removes the one-hot encoding from the node features (first 10 columns)

    Args:
        node_features (torch.Tensor): The node features to remove the one-hot
            encoding from.

    Returns:
        torch.Tensor: The node features with the one-hot encoding removed.
    """
    return node_features[:, 10:]


def remove_one_hot_from_data(data: DenseData) -> DenseData:
    """Removes the one-hot encoding from the node features of the given data.

    Args:
        data (DenseData): The data to remove the one-hot encoding from.

    Returns:
        DenseData: The data with the one-hot encoding removed.
    """
    return DenseData(data.adj_matrix, remove_one_hot(data.x), data.y)


def load_and_split_data(
    folder_names: list[str],
    seed: int = 0,
    eval_size: float = 0.1,
    test_size: float = 0.2,
    show_progress: bool = True,
    data_path: Optional[os.PathLike | str | bytes] = None,
) -> tuple[list[DenseData], list[DenseData], list[DenseData]]:
    """Loads the data from the given folders, splits it into train, eval and
    test sets and returns them."""
    dense_data_list = load_pickle_objects_from_folders(
        folder_names, show_progress=show_progress, data_path=data_path
    )
    train_dense_data, eval_dense_data, test_dense_data = train_eval_test_split(
        dense_data_list,
        seed=seed,
        eval_size=eval_size,
        test_size=test_size,
    )
    return train_dense_data, eval_dense_data, test_dense_data


def load_debug_data():
    path = (
        get_project_path()
        / "data"
        / "difficulty_prediction"
        / "adj_data_list_augmented_benchmark_10machines"
    )
    dense_data = load_pickle_object(path / "0.pkl")
    return [dense_data], [dense_data], [dense_data]


def load_dense_data_dataset(
    load_data_config: dict[str, str | int | float | bool],
    debug: bool = False,
    keep_one_hot: bool = False,
) -> tuple[list[DenseData], list[DenseData], list[DenseData]]:
    if debug:
        return load_debug_data()

    train_data, val_data, test_data = load_and_split_data(**load_data_config)

    if not keep_one_hot:
        train_data = [remove_one_hot_from_data(data) for data in train_data]
        val_data = [remove_one_hot_from_data(data) for data in val_data]
        test_data = [remove_one_hot_from_data(data) for data in test_data]

    return train_data, val_data, test_data
