import os
import random
from job_shop_lib.benchmarking import load_all_benchmark_instances
from gnn_scheduler.configs import Config, ModelConfig
from gnn_scheduler.metrics import Accuracy, F1Score
from gnn_scheduler.utils import get_data_path


TAILLARD15X15_INSTANCES_NAMES = [f"ta{i:02d}" for i in range(1, 11)]
TAILLARD15X15_INSTANCES = [
    instance
    for name, instance in load_all_benchmark_instances().items()
    if name in TAILLARD15X15_INSTANCES_NAMES
]

random_generator = random.Random(42)

# Get all json files under DATA / raw dir with "train" in their name
TRAIN_JSONS = [
    file for file in os.listdir(get_data_path() / "raw") if "train" in file
]
TRAIN_JSONS_WITHOUT_8X8 = [
    file
    for file in os.listdir(get_data_path() / "raw")
    if "train" in file and "8x8" not in file
]
TESTING_JSONS = [
    file for file in os.listdir(get_data_path() / "raw") if "testing" in file
]

# Train JSONS without "10x5" instances
ONLY_10X10_TRAIN_JSONS = [
    file for file in TRAIN_JSONS_WITHOUT_8X8 if "10x5" not in file
]
TRAIN_JSONS_WITHOUT_10X5_AND_8X8 = ONLY_10X10_TRAIN_JSONS + [
    file for file in TRAIN_JSONS_WITHOUT_8X8 if "10x5to10" in file
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
    train_jsons=TRAIN_JSONS_WITHOUT_8X8,
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
    train_jsons=TRAIN_JSONS_WITHOUT_8X8,
    lr=0.0001,
    epochs=100,
    early_stopping_patience=50,
)
EXPERIMENT_6 = Config(  # same than experiment 4 but with AdamW and lr=0.0001
    model_config=ModelConfig(aggregation="max"),
    experiment_name="experiment_6",
    batch_size=512,
    train_jsons=TRAIN_JSONS_WITHOUT_8X8,
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
    train_jsons=TRAIN_JSONS_WITHOUT_8X8,
    lr=0.0001,
    epochs=100,
    early_stopping_patience=22,
)
EXPERIMENT_8 = Config(
    model_config=ModelConfig(no_message_passing=True, use_mlp_encoder=True),
    experiment_name="experiment_5",
    batch_size=512,
    train_jsons=TRAIN_JSONS_WITHOUT_8X8,
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
    train_jsons=TRAIN_JSONS_WITHOUT_8X8,
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
    train_jsons=TRAIN_JSONS_WITHOUT_10X5_AND_8X8,
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
    train_jsons=TRAIN_JSONS_WITHOUT_10X5_AND_8X8,
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
    train_jsons=TRAIN_JSONS_WITHOUT_10X5_AND_8X8,
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
    train_jsons=TRAIN_JSONS_WITHOUT_10X5_AND_8X8,
    lr=0.0001,
    epochs=10_000,
    early_stopping_patience=22,
    store_each_n_steps=31,
)
random_generator.shuffle(TRAIN_JSONS_WITHOUT_8X8)
random_generator.shuffle(TRAIN_JSONS_WITHOUT_10X5_AND_8X8)
EXPERIMENT_16 = Config(
    model_config=ModelConfig(
        aggregation="max",
        num_layers=1,
        hidden_channels=32,
    ),
    experiment_name="experiment16",
    batch_size=512,
    train_jsons=TRAIN_JSONS_WITHOUT_8X8,
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
    train_jsons=TRAIN_JSONS_WITHOUT_10X5_AND_8X8,
    lr=0.0001,
    epochs=10_000,
    early_stopping_patience=220,
    store_each_n_steps=31,
)
random_generator.shuffle(TRAIN_JSONS)
EXPERIMENT_18 = Config(
    model_config=ModelConfig(
        aggregation="max", num_layers=1, hidden_channels=32
    ),
    experiment_name="experiment18",
    batch_size=512,
    train_jsons=TRAIN_JSONS,
    lr=0.0001,
    epochs=10_000,
    early_stopping_patience=100,
    store_each_n_steps=31,
)
EXPERIMENT_19 = Config(
    model_config=ModelConfig(
        aggregation="max", num_layers=1, hidden_channels=32
    ),
    experiment_name="experiment19",
    batch_size=512,
    train_jsons=TRAIN_JSONS,
    lr=0.0001,
    epochs=1000,
    early_stopping_patience=50,
    store_each_n_steps=31,
    use_combined_dataset=True,
)
EXPERIMENT_20 = Config(
    model_config=ModelConfig(
        aggregation="max", num_layers=1, hidden_channels=32
    ),
    experiment_name="experiment20",
    batch_size=512,
    train_jsons=ONLY_10X10_TRAIN_JSONS,
    lr=0.0001,
    epochs=1000,
    early_stopping_patience=50,
    store_each_n_steps=31,
    use_combined_dataset=True,
    combined_dataset_filename="TRAIN_10X10_combined_dataset.pt",
)
EXPERIMENT_21 = Config(
    model_config=ModelConfig(
        aggregation="max", num_layers=1, hidden_channels=32
    ),
    experiment_name="experiment21",
    batch_size=512,
    train_jsons=ONLY_10X10_TRAIN_JSONS,
    lr=0.0001,
    epochs=1000,
    early_stopping_patience=50,
    store_each_n_steps=99,
    use_combined_dataset=True,
    combined_dataset_filename="TRAIN_10X10_subset99_combined_dataset.pt",
)
EXPERIMENT_22 = Config(
    model_config=ModelConfig(
        aggregation="max",
        num_layers=1,
        hidden_channels=32,
        use_batch_norm=False,
    ),
    experiment_name="experiment22",
    batch_size=512,
    train_jsons=ONLY_10X10_TRAIN_JSONS,
    lr=0.0001,
    epochs=1000,
    early_stopping_patience=50,
    store_each_n_steps=99,
    use_combined_dataset=True,
    combined_dataset_filename="TRAIN_10X10_subset99_combined_dataset.pt",
)
EXPERIMENT_23 = Config(
    model_config=ModelConfig(
        aggregation="max",
        num_layers=2,  # 2 layers
        hidden_channels=32,
        use_batch_norm=False,
    ),
    experiment_name="experiment23",
    batch_size=512,
    train_jsons=ONLY_10X10_TRAIN_JSONS,
    lr=0.0001,
    epochs=1000,
    early_stopping_patience=50,
    store_each_n_steps=99,
    use_combined_dataset=True,
    combined_dataset_filename="TRAIN_10X10_subset99_combined_dataset.pt",
)
EXPERIMENT_24 = Config(
    model_config=ModelConfig(
        aggregation="max",
        num_layers=2,
        hidden_channels=32,
        use_batch_norm=False,
    ),
    experiment_name="experiment24",
    batch_size=512,
    train_jsons=ONLY_10X10_TRAIN_JSONS,
    lr=0.0001,
    epochs=1000,
    early_stopping_patience=50,
    store_each_n_steps=31,
    use_combined_dataset=True,
    combined_dataset_filename="TRAIN_10X10_combined_dataset.pt",
)
EXPERIMENT_25 = Config(
    model_config=ModelConfig(
        aggregation="max",
        num_layers=3,  # 3 layers
        hidden_channels=32,
        use_batch_norm=False,
    ),
    experiment_name="experiment25",
    batch_size=512,
    train_jsons=ONLY_10X10_TRAIN_JSONS,
    lr=0.0001,
    epochs=1000,
    early_stopping_patience=50,
    store_each_n_steps=31,
    use_combined_dataset=True,
    combined_dataset_filename="TRAIN_10X10_combined_dataset.pt",
    eval_instances=TAILLARD15X15_INSTANCES,
    val_dataset_filename="eval10to15x5to10.json",
)
EXPERIMENT_26 = Config(
    model_config=ModelConfig(
        aggregation="mean",  # mean aggregation
        num_layers=3,
        hidden_channels=32,
        use_batch_norm=False,
    ),
    experiment_name="experiment26",
    batch_size=512,
    train_jsons=ONLY_10X10_TRAIN_JSONS,
    lr=0.0001,
    epochs=1000,
    early_stopping_patience=50,
    store_each_n_steps=31,
    use_combined_dataset=True,
    combined_dataset_filename="TRAIN_10X10_combined_dataset.pt",
)
EXPERIMENT_27 = Config(
    model_config=ModelConfig(
        aggregation="max",  # max aggregation
        num_layers=2,  # 2 layers
        hidden_channels=64,  # 64 hidden channels
        use_batch_norm=False,
    ),
    experiment_name="experiment27",
    batch_size=512,
    train_jsons=ONLY_10X10_TRAIN_JSONS,
    lr=0.0001,
    epochs=1000,
    early_stopping_patience=50,
    store_each_n_steps=31,
    use_combined_dataset=True,
    combined_dataset_filename="TRAIN_10X10_combined_dataset.pt",
)
EXPERIMENT_28 = Config(
    model_config=ModelConfig(
        aggregation="max",
        num_layers=2,
        hidden_channels=32,
        use_batch_norm=False,
    ),
    experiment_name="experiment28",
    batch_size=512,
    train_jsons=TRAIN_JSONS,  # all train
    lr=0.0001,
    epochs=1000,
    early_stopping_patience=15,
    store_each_n_steps=31,
    use_combined_dataset=True,
)
EXPERIMENT_29 = Config(
    model_config=ModelConfig(
        aggregation="max",
        num_layers=2,
        hidden_channels=32,
        use_batch_norm=False,
    ),
    experiment_name="experiment29",
    batch_size=512,
    train_jsons=TRAIN_JSONS,  # all train
    lr=0.0001,
    epochs=1000,
    early_stopping_patience=15,
    store_each_n_steps=31,
    use_combined_dataset=True,
    val_dataset_filename="eval10to15x5to10.json",
)

EXPERIMENT_30 = Config(
    model_config=ModelConfig(
        aggregation="max",
        num_layers=2,
        hidden_channels=48,
        use_batch_norm=False,
    ),
    experiment_name="experiment30",
    batch_size=512,
    train_jsons=TRAIN_JSONS,  # all train
    lr=0.0001,
    epochs=1000,
    early_stopping_patience=15,
    store_each_n_steps=31,
    use_combined_dataset=True,
    eval_instances=TAILLARD15X15_INSTANCES,
    val_dataset_filename="eval10to15x5to10.json",
)

EXPERIMENT_31 = Config(
    model_config=ModelConfig(
        aggregation="max",
        num_layers=2,
        hidden_channels=32,
        use_batch_norm=False,
        gnn_type="HGATV2",
    ),
    experiment_name="experiment31",
    batch_size=512,
    train_jsons=TRAIN_JSONS,
    lr=0.0001,
    epochs=1000,
    early_stopping_patience=20,
    store_each_n_steps=31,
    use_combined_dataset=True,
    eval_instances=TAILLARD15X15_INSTANCES,
    val_dataset_filename="eval10to15x5to10.json",
    primary_val_key="eval10to15x5to10",
    combined_dataset_filename="TRAIN_combined_dataset.pt",
)
EXPERIMENT_32 = Config(
    model_config=ModelConfig(
        aggregation="max",
        num_layers=1,
        hidden_channels=32,
        use_batch_norm=False,
        gnn_type="HGATV2",
        edge_dropout=0.3,
    ),
    experiment_name="experiment32",
    batch_size=512,
    train_jsons=TRAIN_JSONS,
    lr=0.0001,
    epochs=1000,
    early_stopping_patience=20,
    store_each_n_steps=31,
    use_combined_dataset=True,
    eval_instances=TAILLARD15X15_INSTANCES,
    val_dataset_filename="eval10to15x5to10.json",
    primary_val_key="eval10to15x5to10",
    combined_dataset_filename="TRAIN_combined_dataset.pt",
)

EXPERIMENT_33 = Config(
    model_config=ModelConfig(
        aggregation="max",
        num_layers=1,
        hidden_channels=32,
        use_batch_norm=False,
        gnn_type="HGATV2",
        edge_dropout=0.3,
    ),
    experiment_name="experiment33",
    batch_size=512,
    train_jsons=TRAIN_JSONS,
    lr=0.0001,
    epochs=1000,
    early_stopping_patience=20,
    store_each_n_steps=31,
    use_combined_dataset=True,
    eval_instances=TAILLARD15X15_INSTANCES,
    val_dataset_filename="eval10to15x5to10.json",
    primary_val_key="eval10to15x5to10",
    combined_dataset_filename="TRAIN_combined_dataset.pt",
    allow_operation_reservation=True,
)

EXPERIMENT_34 = Config(
    model_config=ModelConfig(
        aggregation="max",
        num_layers=1,
        hidden_channels=32,
        use_batch_norm=False,
        gnn_type="HGATV2",
        edge_dropout=0.3,
    ),
    experiment_name="experiment34",
    batch_size=512,
    train_jsons=ONLY_10X10_TRAIN_JSONS,
    lr=0.0001,
    epochs=1000,
    early_stopping_patience=20,
    store_each_n_steps=31,
    use_combined_dataset=True,
    eval_instances=TAILLARD15X15_INSTANCES,
    val_dataset_filename="eval10to15x5to10.json",
    primary_val_key="eval10to15x5to10",
    combined_dataset_filename="TRAIN_10X10_combined_dataset.pt",
    allow_operation_reservation=True,
)

EXPERIMENT_35 = Config(
    model_config=ModelConfig(
        use_batch_norm=False,
        no_message_passing=True,
    ),
    experiment_name="experiment35",
    batch_size=512,
    train_jsons=TRAIN_JSONS,  # all train
    lr=0.0001,
    epochs=1000,
    early_stopping_patience=15,
    store_each_n_steps=31,
    use_combined_dataset=True,
    eval_instances=TAILLARD15X15_INSTANCES,
    val_dataset_filename="eval10to15x5to10.json",
)

EXPERIMENT_36 = Config(
    model_config=ModelConfig(
        aggregation="max",
        num_layers=2,
        hidden_channels=32,
        use_batch_norm=False,
        gnn_type="HGATV2",
        edge_dropout=0.25,
    ),
    experiment_name="experiment36",
    batch_size=512,
    train_jsons=TRAIN_JSONS,
    lr=0.0001,
    epochs=1000,
    early_stopping_patience=100,
    store_each_n_steps=11,
    use_combined_dataset=True,
    eval_instances=TAILLARD15X15_INSTANCES,
    val_dataset_filename="eval10to15x5to10.json",
    primary_val_key="eval10to15x5to10",
    combined_dataset_filename="TRAIN_combined_dataset_subset11.pt",
    allow_operation_reservation=True,
)
EXPERIMENT_37 = Config(
    model_config=ModelConfig(
        aggregation="max",
        num_layers=2,
        hidden_channels=32,
        use_batch_norm=False,
        gnn_type="HGATV2",
        edge_dropout=0.2,
    ),
    experiment_name="experiment37",
    batch_size=512,
    train_jsons=TRAIN_JSONS,
    lr=0.0001,
    epochs=1000,
    early_stopping_patience=100,
    store_each_n_steps=17,
    use_combined_dataset=True,
    eval_instances=TAILLARD15X15_INSTANCES,
    val_dataset_filename="eval10to15x5to10.json",
    primary_val_key="eval10to15x5to10",
    combined_dataset_filename="TRAIN_combined_dataset_subset11.pt",
    allow_operation_reservation=True,
)
EXPERIMENT_38 = Config(
    model_config=ModelConfig(
        aggregation="max",
        num_layers=2,
        hidden_channels=32,
        use_batch_norm=False,
        gnn_type="HGATV2",
        edge_dropout=0.1,
    ),
    experiment_name="experiment38",
    batch_size=512,
    train_jsons=TRAIN_JSONS,
    lr=0.0001,
    epochs=1000,
    early_stopping_patience=100,
    store_each_n_steps=11,
    use_combined_dataset=True,
    eval_instances=TAILLARD15X15_INSTANCES,
    val_dataset_filename="eval10to15x5to10.json",
    primary_val_key="eval10to15x5to10",
    combined_dataset_filename="TRAIN_combined_dataset_subset11.pt",
    allow_operation_reservation=True,
)
