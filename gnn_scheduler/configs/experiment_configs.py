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
