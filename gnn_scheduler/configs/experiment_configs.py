import os
import random

random_generator = random.Random(42)
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

# Train JSONS without "10x5" instances
ONLY_10X10_TRAIN_JSONS = [file for file in TRAIN_JSONS if "10x5" not in file]
TRAIN_JSONS_WITHOUT_10X5 = ONLY_10X10_TRAIN_JSONS + [
    file for file in TRAIN_JSONS if "10x5to10" in file
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
    batch_size=512,
    train_jsons=TRAIN_JSONS,
    lr=0.0005,
    epochs=100,
    early_stopping_patience=50,
)
TESTING_CONFIG = Config(
    model_config=ModelConfig(aggregation="max"),
    experiment_name="debugging_dataset_manager",
    batch_size=256,
    train_jsons=TESTING_JSONS,
    lr=0.0005,
    epochs=2,
)
EXPERIMENT_5 = Config(
    model_config=ModelConfig(no_message_passing=True),
    experiment_name="experiment_5",
    batch_size=512,
    train_jsons=TRAIN_JSONS,
    lr=0.0001,
    epochs=100,
    early_stopping_patience=50,
)
EXPERIMENT_6 = Config(  # same than experiment 4 but with AdamW and lr=0.0001
    model_config=ModelConfig(aggregation="max"),
    experiment_name="experiment_6",
    batch_size=512,
    train_jsons=TRAIN_JSONS,
    lr=0.0001,
    epochs=100,
    early_stopping_patience=22,
)
EXPERIMENT_7 = Config(  # same than experiment 4 but with AdamW and lr=0.0001
    model_config=ModelConfig(
        aggregation="max", num_layers=1, hidden_channels=32
    ),
    experiment_name="experiment_7",
    batch_size=512,
    train_jsons=TRAIN_JSONS,
    lr=0.0001,
    epochs=100,
    early_stopping_patience=22,
)
EXPERIMENT_8 = Config(
    model_config=ModelConfig(no_message_passing=True, use_mlp_encoder=True),
    experiment_name="experiment_5",
    batch_size=512,
    train_jsons=TRAIN_JSONS,
    lr=0.0001,
    epochs=30,
    early_stopping_patience=22,
)
EXPERIMENT_9 = Config(
    model_config=ModelConfig(
        aggregation="max", num_layers=1, hidden_channels=32
    ),
    experiment_name="experiment_9",
    batch_size=512,
    train_jsons=TRAIN_JSONS,
    lr=0.0001,
    epochs=10_000,
    early_stopping_patience=22,
    n_batches_per_epoch=100,
)
EXPERIMENT_10 = Config(
    model_config=ModelConfig(
        aggregation="max", num_layers=1, hidden_channels=32
    ),
    experiment_name="experiment10",
    batch_size=256,
    train_jsons=TRAIN_JSONS_WITHOUT_10X5,
    lr=0.0001,
    epochs=10_000,
    early_stopping_patience=22,
    n_batches_per_epoch=10,
)
EXPERIMENT_11 = Config(
    model_config=ModelConfig(
        aggregation="max", num_layers=1, hidden_channels=32
    ),
    experiment_name="experiment11",
    batch_size=256,
    train_jsons=TRAIN_JSONS_WITHOUT_10X5,
    lr=0.0001,
    epochs=10_000,
    early_stopping_patience=22,
    n_batches_per_epoch=10,
    store_each_n_steps=31,
)
EXPERIMENT_12 = Config(
    model_config=ModelConfig(
        aggregation="max", num_layers=1, hidden_channels=32, edge_dropout=0.4
    ),
    experiment_name="experiment12",
    batch_size=512,
    train_jsons=ONLY_10X10_TRAIN_JSONS,
    lr=0.0001,
    epochs=10_000,
    early_stopping_patience=22,
    # store_each_n_steps=31,
    n_batches_per_epoch=500,
)
EXPERIMENT_13 = Config(
    model_config=ModelConfig(
        aggregation="max", num_layers=1, hidden_channels=32,
    ),
    experiment_name="experiment13",
    batch_size=512,
    train_jsons=ONLY_10X10_TRAIN_JSONS,
    lr=0.0001,
    epochs=10_000,
    early_stopping_patience=22,
    store_each_n_steps=31,
)
EXPERIMENT_14 = Config(
    model_config=ModelConfig(
        aggregation="max",
        num_layers=1,
        hidden_channels=32,
    ),
    experiment_name="experiment14",
    batch_size=512,
    train_jsons=TRAIN_JSONS_WITHOUT_10X5,
    lr=0.0001,
    epochs=10_000,
    early_stopping_patience=22,
    store_each_n_steps=31,
)
EXPERIMENT_15 = Config(
    model_config=ModelConfig(
        aggregation="max",
        num_layers=1,
        hidden_channels=32,
    ),
    experiment_name="experiment15",
    batch_size=512,
    train_jsons=TRAIN_JSONS_WITHOUT_10X5,
    lr=0.0001,
    epochs=10_000,
    early_stopping_patience=22,
    store_each_n_steps=31,
)
random_generator.shuffle(TRAIN_JSONS)
random_generator.shuffle(TRAIN_JSONS_WITHOUT_10X5)
EXPERIMENT_16 = Config(
    model_config=ModelConfig(
        aggregation="max",
        num_layers=1,
        hidden_channels=32,
    ),
    experiment_name="experiment16",
    batch_size=512,
    train_jsons=TRAIN_JSONS,
    lr=0.0001,
    epochs=10_000,
    early_stopping_patience=220,
    store_each_n_steps=31,
)
EXPERIMENT_17 = Config(
    model_config=ModelConfig(
        aggregation="max", num_layers=1, hidden_channels=32
    ),
    experiment_name="experiment17",
    batch_size=512,
    train_jsons=TRAIN_JSONS_WITHOUT_10X5,
    lr=0.0001,
    epochs=10_000,
    early_stopping_patience=220,
    store_each_n_steps=31,
)
