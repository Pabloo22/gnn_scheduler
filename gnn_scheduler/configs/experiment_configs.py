import os

from gnn_scheduler.configs import Config, ModelConfig
from gnn_scheduler.metrics import Accuracy, F1Score
from gnn_scheduler.utils import get_data_path

# Get all json files under DATA / raw dir with "train" in their name
TRAIN_JSONS = [
    file for file in os.listdir(get_data_path() / "raw") if "train" in file
]
TESTING_JSONS = [
    file for file in os.listdir(get_data_path() / "raw") if "testing" in file
]

DEFAULT_CONFIG = Config()
EXPERIMENT_1 = Config(
    experiment_name="experiment_1",
    metrics=[Accuracy(), F1Score()],
)
EXPERIMENT_2 = Config(
    model_config=ModelConfig(aggregation="max"),
    experiment_name="experiment_2",
    batch_size=256
)
EXPERIMENT_3 = Config(
    model_config=ModelConfig(aggregation="max"),
    experiment_name="experiment_3",
    batch_size=256,
    train_jsons="instances10x10_train_1.json",
    processed_filenames_prefix_train="instances_train10x10_1",
    lr=0.0001,
    epochs=100,
)
EXPERIMENT_4 = Config(
    model_config=ModelConfig(aggregation="max"),
    experiment_name="experiment_4",
    batch_size=256,
    train_jsons=TRAIN_JSONS,
    processed_filenames_prefix_train="instances_train10x10_2",
    lr=0.0005,
    epochs=100,
)
TESTING_CONFIG = Config(
    model_config=ModelConfig(aggregation="max"),
    experiment_name="debugging_dataset_manager",
    batch_size=5,
    train_jsons=TESTING_JSONS,
    processed_filenames_prefix_train="instances_train10x10_2",
    lr=0.0005,
    epochs=2,
)
