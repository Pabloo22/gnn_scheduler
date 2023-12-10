"""
.
"""
import torch
import tqdm
from torch import optim
import wandb

from gnn_scheduler.difficulty_prediction import ExperimentConfig
from gnn_scheduler.gnns.models import RelationalGCNRegressor
from gnn_scheduler import get_project_path


CONFIG_FOLDER = (
    get_project_path() / "gnn_scheduler/difficulty_prediction/config"
)
MODELS_PATH = get_project_path() / "models/difficulty_prediction/model"


def train(experiment_config: ExperimentConfig):
    """Trains a relational GCN regressor on the difficulty prediction task."""

    # Unpack the configurations
    load_data_config = experiment_config.load_data_config
    model_config = experiment_config.model_config
    training_config = experiment_config.training_config

    # Your training logic here, utilizing the above configs
    model = RelationalGCNRegressor(
        in_features=load_data_config.in_features,
        conv_units=model_config.conv_units,
        aggregation_units=model_config.aggregation_units,
        dropout_rate=model_config.dropout_rate,
        leaky_relu_slope=model_config.leaky_relu_slope,
    )
    model.train()
    
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    
    # Training loop
    
    

def main():
    # Initialize wandb
    wandb.init(project="difficulty_prediction")

    # Load experiment configuration
    experiment_config = ExperimentConfig(CONFIG_FOLDER)

    # Check if running as part of wandb sweep
    if wandb.run.sweep_id is not None:
        # If part of a sweep, override experiment config with wandb config
        experiment_config = wandb.config

    train(experiment_config)

    wandb.finish()


if __name__ == "__main__":
    print(ExperimentConfig(CONFIG_FOLDER))
