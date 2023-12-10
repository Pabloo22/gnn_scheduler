from gnn_scheduler.jssp import (
    load_pickle_instances_from_folders,
    AdjData,
    instance_to_adj_data,
    diff_pred_node_features_creators,
)
from gnn_scheduler.gnns import train_eval_test_split


def load_data(
    folder_names: list[str],
    seed: int = 0,
    eval_size: float = 0.1,
    test_size: float = 0.2,
) -> tuple[AdjData, AdjData, AdjData]:
    """Loads the data from the given folders as AdjData objects.

    Args:
        folder_names (list[str]): the names of the folders containing the
            instances
        seed (int, optional): the seed for the train test split. Defaults to 0.
        eval_size (float, optional): the proportion of instances to use for
            evaluation within the train set. Defaults to 0.1.
        test_size (float, optional): the proportion of instances to use for
            testing. Defaults to 0.2.


    Returns:
        tuple[AdjData, AdjData, AdjData]: the train, eval and test sets.
    """
    instances = load_pickle_instances_from_folders(folder_names)

    node_feature_creators = diff_pred_node_features_creators()
    adj_data = [
        instance_to_adj_data(instance, node_feature_creators)
        for instance in instances
    ]
    train, evaluation, test = train_eval_test_split(
        adj_data, seed=seed, eval_size=eval_size, test_size=test_size
    )
    return train, evaluation, test