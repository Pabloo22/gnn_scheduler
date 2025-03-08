from gnn_scheduler.configs import Config, ModelConfig
from gnn_scheduler.metrics import Accuracy, F1Score


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
    train_json="instances10x10_train_1.json",
    processed_filenames_prefix_train="instances_train10x10_1",
    lr=0.0001,
    epochs=100,
)
