from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader

from gnn_scheduler.jssp import load_all_from_benchmark
from gnn_scheduler.data.preprocessing import (preprocess_graphs,
                                              MachineLoad,
                                              OperationIndex,
                                              Duration,
                                              JobLoad,
                                              normalize_optimum,
                                              )


def create_data_loader(batch_size: int = 32):
    instances = load_all_from_benchmark(max_jobs=20, max_machines=10)
    graphs = [instance.disjunctive_graph for instance in instances]
    graphs = preprocess_graphs(graphs,
                               node_feature_creators=[OperationIndex(),
                                                      Duration(),
                                                      Duration(normalize_with="job"),
                                                      Duration(normalize_with="machine"),
                                                      JobLoad(),
                                                      MachineLoad(),
                                                      ],
                               remove_nodes=["S", "T"],
                               keep_old_features=True,
                               copy=True,
                               )

    # Use from_networkx to convert the graphs to PyG format.
    data_list = [from_networkx(graph) for graph in graphs]

    # Add the target to the data.
    for instance, data in zip(instances, data_list):
        data.y = normalize_optimum(instance.optimum, instance.disjunctive_graph)

    loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)
    num_node_features = data_list[0].x.shape[1]
    return loader, num_node_features


if __name__ == "__main__":
    create_data_loader()
    