from dataclasses import dataclass, field
from gnn_scheduler.model import HeteroMetadata
from gnn_scheduler.metrics import Accuracy, Precision, Recall, F1Score, Metric


@dataclass
class ModelConfig:
    metadata: HeteroMetadata = field(
        default_factory=lambda: HeteroMetadata(
            node_types=["operation", "machine"]
        )
    )
    in_channels_dict: dict[str, int] = field(
        default_factory=lambda: {"operation": 8, "machine": 4}
    )
    initial_node_features_dim: int = 32
    sigma: float = 1.0
    hidden_channels: int = 64
    num_layers: int = 3
    use_batch_norm: bool = True
    aggregation: str = "sum"

    def to_dict(self):
        return {
            "metadata": self.metadata,
            "in_channels_dict": self.in_channels_dict,
            "initial_node_features_dim": self.initial_node_features_dim,
            "sigma": self.sigma,
            "hidden_channels": self.hidden_channels,
            "num_layers": self.num_layers,
            "use_batch_norm": self.use_batch_norm,
            "aggregation": self.aggregation,
        }


@dataclass
class Config:
    model_config: ModelConfig = field(default_factory=ModelConfig)
    lr: float = 0.001
    num_chunks_train: int = 100
    max_chunks_in_memory: int = 100
    metrics: list[Metric] = field(
        default_factory=lambda: [Accuracy(), Precision(), Recall(), F1Score()]
    )
    epochs: int = 50
    experiment_name: str = "debugging"
    grad_clip_val: float = 1.0
    early_stopping_patience: int = 10
    primary_val_key: str = "instances10x10_eval"
    batch_size: int = 128
    scheduler_: dict | None = None
    train_json: str = "small_random_instances_0.json"
    processed_filenames_prefix_train: str = "instances_train10x10"
