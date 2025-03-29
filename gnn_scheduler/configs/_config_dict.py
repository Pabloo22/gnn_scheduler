from dataclasses import dataclass, field
from job_shop_lib import JobShopInstance
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
    no_message_passing: bool = False
    use_mlp_encoder: bool = False
    edge_dropout: float = 0.0
    gnn_type: str = "HGIN"

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
            "no_message_passing": self.no_message_passing,
            "use_mlp_encoder": self.use_mlp_encoder,
            "edge_dropout": self.edge_dropout,
            "gnn_type": self.gnn_type,
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
    train_jsons: str | list[str] = "small_random_instances_0.json"
    processed_filenames_prefix_train: str = "instances_train10x10"
    n_batches_per_epoch: int | None = None
    store_each_n_steps: int = 1
    use_combined_dataset: bool = False
    combined_dataset_filename: str = "TRAIN_combined_dataset.pt"
    val_dataset_filename: str = "instances10x10_eval_0.json"
    eval_instances: list[JobShopInstance] | None = None
    allow_operation_reservation: bool = False
    neighbor_sampling: int | None = None
