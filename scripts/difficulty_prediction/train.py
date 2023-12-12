"""
Script to train a relational GCN regressor on the difficulty prediction task.
This script is no longer used, since the GAN project is not going to be
continued for now.
"""
import random

import torch
import tqdm
from torch import optim
import wandb

from gnn_scheduler.difficulty_prediction import (
    ExperimentConfig,
    load_and_split_data,
)
from gnn_scheduler.gnns.models.dense_matrix import (
    RelationalGCNRegressor,
    Discriminator,
)

from gnn_scheduler.jssp.graphs import AdjData
from gnn_scheduler.jssp import load_pickle_object
from gnn_scheduler import get_project_path


CONFIG_FOLDER = get_project_path() / "configs/difficulty_prediction/"
MODELS_PATH = get_project_path() / "models/difficulty_prediction/"

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


def random_name() -> str:
    """Generates a random name for a run.

    Returns:
        str: The random name.
    """
    return "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=5))


def set_all_seeds(seed: int):
    """Sets all seeds to the given value.

    Args:
        seed (int): The seed to set.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(
    model: torch.nn.Module,
    data: list[AdjData],
    show_progress: bool = True,
) -> float:
    """Evaluates the model on the given data.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data (list[AdjData]): The data to evaluate on.
        show_progress (bool, optional): Whether to show a progress bar.

    Returns:
        float: The mean squared error of the model on the given data.
    """
    model.eval()

    criterion = torch.nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for adj_data in tqdm.tqdm(
            data, disable=not show_progress, desc="Evaluation"
        ):
            # Move data to device
            node_features = adj_data.x.to(device)
            adj_matrices = adj_data.adj_matrix.to(device)
            label = torch.tensor([[adj_data.y]]).to(device)

            output = model(
                node_features=node_features, adj_matrices=adj_matrices
            )
            loss = criterion(output, label)
            total_loss += loss.item()
    return total_loss / len(data)


def train(experiment_config: dict[str, dict]):
    """Trains a relational GCN regressor on the difficulty prediction task."""
    using_wandb = wandb.run is not None
    if using_wandb:
        wandb.config.update(experiment_config)

    # Unpack the configurations
    load_data_config = experiment_config["load_data_config"]
    model_config = experiment_config["model_config"]
    training_config = experiment_config["training_config"]

    # train_data, val_data, test_data = load_and_split_data(
    #     folder_names=load_data_config["folder_names"],
    #     seed=load_data_config["seed"],
    #     eval_size=load_data_config["eval_size"],
    #     test_size=load_data_config["test_size"],
    #     show_progress=training_config["show_progress"],
    # )
    path = (
        get_project_path()
        / "data"
        / "adj_data_list_augmented_benchmark_10machines"
        / "0.pkl"
    )
    train_data = [load_pickle_object(path)]
    val_data = []
    test_data = []

    run_name = wandb.run.name if using_wandb else random_name()
    set_all_seeds(load_data_config["seed"])

    # Your training logic here, utilizing the above configs
    # model = RelationalGCNRegressor(
    #     in_features=model_config["in_features"],
    #     conv_units=model_config["conv_units"],
    #     aggregation_units=model_config["aggregation_units"],
    #     dropout_rate=model_config["dropout_rate"],
    #     leaky_relu_slope=model_config["leaky_relu_slope"],
    # )
    model = Discriminator(
        conv_dim=model_config["conv_dim"],
        m_dim=model_config["m_dim"],
        b_dim=model_config["b_dim"],
        dropout_rate=model_config["dropout_rate"],
    )

    model.to(device)
    if using_wandb:
        # Watch the model to log gradients and parameters
        wandb.watch(model, log_freq=1)

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )

    # Training loop
    for epoch in range(training_config["n_epochs"]):
        model.train()
        total_loss = 0.0
        for adj_data in tqdm.tqdm(
            train_data,
            disable=not training_config["show_progress"],
            desc=f"Epoch {epoch}",
        ):
            # Move data to device
            node_features = adj_data.x.to(device)
            adj_matrices = adj_data.adj_matrix.to(device)
            label = torch.tensor([[adj_data.y]]).to(device)

            optimizer.zero_grad()
            output = model(
                node_features=node_features, adj_matrices=adj_matrices
            )
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if using_wandb:
                wandb.log({"loss": loss.item()})
                # Log the prediction of the model
                wandb.log({"prediction": output.item()})

        # val_loss = evaluate(
        #     model,
        #     val_data,
        #     show_progress=training_config["show_progress"],
        # )
        # if using_wandb:
        #     wandb.log({"avg_train_loss": total_loss / len(train_data)})
        #     wandb.log({"val_loss": val_loss})
        # print(f"Validation loss: {val_loss}")

        # Save the model
        # torch.save(
        #     model.state_dict(),
        #     MODELS_PATH / f"gcn_regressor_{run_name}_{epoch}.pt",
        # )

    # Evaluate the model on the test set
    # test_loss = evaluate(
    #     model,
    #     test_data,
    #     show_progress=training_config["show_progress"],
    # )
    # print(f"Test loss: {test_loss}")
    # if using_wandb:
    #     wandb.log({"test_loss": test_loss})


def main():
    # Initialize wandb
    wandb.init(project="difficulty_prediction")

    # Load experiment configuration
    experiment_config = ExperimentConfig(CONFIG_FOLDER).to_dict()

    # Check if running as part of wandb sweep
    if wandb.run.sweep_id is not None:
        # If part of a sweep, override experiment config with wandb config
        experiment_config = wandb.config

    train(experiment_config)

    wandb.finish()


if __name__ == "__main__":
    main()
