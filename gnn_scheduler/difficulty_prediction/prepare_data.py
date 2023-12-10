from gnn_scheduler.jssp import (
    JobShopInstance,
    load_pickle_instances_from_folders,
)
from gnn_scheduler.jssp.graphs import (
    NodeFeatureCreator,
    JobID,
    OperationIndex,
    OneHotEncoding,
    AdjData,
    disjunctive_graph_to_tensors,
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
) -> AdjData:
    """Returns the node features and adjacency matrices of a job-shop instance.

    Args:
        instance (JobShopInstance): the instance.
        node_feature_creators (list[NodeFeatureCreator]): the node feature
            creators to use.
        copy (bool, optional): whether to copy the graph before preprocessing.

    Returns:
        AdjData: the node features and adjacency matrices
    """
    y = get_difficulty_score(instance)
    node_features, adj_matrices = disjunctive_graph_to_tensors(
        instance.disjunctive_graph,
        node_feature_creators=node_feature_creators,
        copy=copy,
    )

    return AdjData(adj_matrix=adj_matrices, x=node_features, y=y)


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


if __name__ == "__main__":
    print(help(get_difficulty_score))
