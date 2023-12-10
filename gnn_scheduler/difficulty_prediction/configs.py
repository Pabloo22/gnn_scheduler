import os
from dataclasses import dataclass

from gnn_scheduler.gnns.training import load_yaml


@dataclass
class LoadDataConfig:
    folder_names: list[str]
    path: str | None = None
    seed: int = 0
    eval_size: float = 0.1
    test_size: float = 0.2


@dataclass
class ModelConfig:
    in_features: int
    conv_units: list[int]
    aggregation_units: int
    dropout_rate: float = 0.0
    leaky_relu_slope: float = 0.1


@dataclass
class TrainingConfig:
    n_epochs: int
    optimizer: str = "adam"
    learning_rate: float = 0.001
    batch_size: int = 1
    weight_decay: float = 0.0001
    shuffle: bool = True
    show_progress: bool = True


class ExperimentConfig:
    def __init__(
        self, config_folder: os.PathLike | str | bytes
    ):
        load_data_path = os.path.join(config_folder, "load_data.yaml")
        model_path = os.path.join(config_folder, "model.yaml")
        training_path = os.path.join(config_folder, "training.yaml")

        self.load_data_config = load_yaml(load_data_path, LoadDataConfig)
        self.model_config = load_yaml(model_path, ModelConfig)
        self.training_config = load_yaml(training_path, TrainingConfig)

    def to_dict(self):
        return {
            "load_data_config": self.load_data_config.__dict__,
            "model_config": self.model_config.__dict__,
            "training_config": self.training_config.__dict__,
        }

    def __repr__(self):
        return (
            f"ExperimentConfig(load_data_config={self.load_data_config}, "
            f"model_config={self.model_config}, "
            f"training_config={self.training_config})"
        )


if __name__ == "__main__":
    from gnn_scheduler import get_project_path

    CONFIG_FOLDER = (
        get_project_path() / "gnn_scheduler/difficulty_prediction/config"
    )
    experiment_config = ExperimentConfig(CONFIG_FOLDER)
    print(experiment_config)
    print(experiment_config.to_dict())
