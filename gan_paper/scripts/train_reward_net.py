"""
Script to train a relational GCN regressor on the difficulty prediction task.
"""
import yaml

import wandb

# from gnn_scheduler.jssp import load_pickle_object
from gnn_scheduler import get_project_path
from gnn_scheduler.gan import RewardNetworkTrainer

CONFIG_FOLDER = get_project_path() / "configs/difficulty_prediction/"
CONFIG_FILE = "adj_config.yaml"
MODELS_PATH = get_project_path() / "models/difficulty_prediction/"

USE_WANDB = True

def train(experiment_config: dict):
    trainer = RewardNetworkTrainer(experiment_config, use_wandb=USE_WANDB)
    
    trainer.compile()
    trainer.fit()
    

def main():
    # Initialize wandb
    if USE_WANDB:
        wandb.init(project="difficulty_prediction")

    # Load experiment configuration
    path = CONFIG_FOLDER / CONFIG_FILE
    with open(path, "r", encoding="utf-8") as f:
        experiment_config = yaml.safe_load(f)

    # Check if running as part of wandb sweep
    if USE_WANDB and wandb.run.sweep_id is not None:
        # If part of a sweep, override experiment config with wandb config
        experiment_config = wandb.config

    train(experiment_config)

    if USE_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
