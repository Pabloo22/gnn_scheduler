import torch
from torch.optim import Adam
from torch.nn import MSELoss
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader

import wandb

from gnn_scheduler.experiments.difficulty_prediction import (
    GCNRegression
)
from gnn_scheduler.jssp import load_all_from_benchmark
from gnn_scheduler.jssp.graphs import (preprocess_graphs,
                                              MachineLoad,
                                              OperationIndex,
                                              Duration,
                                              JobLoad,
                                              normalize_optimum,
                                              )

BATCH_SIZE = 16
LEARNING_RATE = 0.01
EPOCHS = 100


def create_data_loader(batch_size: int = 32):
    instances = load_all_from_benchmark(if_has_optimum=True)
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


def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        # Move data to GPU
        data = data.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y.view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    wandb.init(project="jssp-makespan-prediction")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    loader, num_node_features = create_data_loader(batch_size=BATCH_SIZE)
    model = GCNRegression(num_node_features).to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = MSELoss()
    
    wandb.config = {
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
    }
    # Optionally, watch the model
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Training the model
    for epoch in range(EPOCHS):
        loss = train(model, loader, optimizer, criterion, device)
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}')

        # Log training metrics
        wandb.log({"epoch": epoch, "loss": loss})

    wandb.finish()


if __name__ == "__main__":
    main()
