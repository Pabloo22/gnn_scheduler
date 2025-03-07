import torch
from torch_geometric.loader import DataLoader
from gnn_scheduler.model import ResidualSchedulingGNN
from gnn_scheduler.data import JobShopDataset

from gnn_scheduler.trainer import Trainer
from gnn_scheduler.configs import Config


def _main(config: Config):
    # Create or load your model
    model = ResidualSchedulingGNN(
        **config.model_config.to_dict(),
    )
    train_dataset = JobShopDataset(
        num_chunks=config.num_chunks_train,
        max_chunks_in_memory=config.max_chunks_in_memory,
    )
    val_dataset_10x10 = JobShopDataset(
        num_chunks=1,
        raw_filename="instances10x10_eval_0.json",
        processed_filenames_prefix="instances10x10_eval",
    )
    val_dataset_5x5 = JobShopDataset(
        num_chunks=1,
        raw_filename="instances5x5_eval_0.json",
        processed_filenames_prefix="instances5x5_eval",
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_dataloader_10x10 = DataLoader(
        val_dataset_10x10, batch_size=config.batch_size
    )
    val_dataloader_5x5 = DataLoader(
        val_dataset_5x5, batch_size=config.batch_size
    )

    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Create loss function
    criterion = torch.nn.BCEWithLogitsLoss()

    # Set up learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="max", factor=0.5, patience=5
    # )

    # Initialize trainer with multiple metrics
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloaders={
            "instances10x10_eval": val_dataloader_10x10,
            "instances5x5_eval": val_dataloader_5x5,
        },
        optimizer=optimizer,
        criterion=criterion,
        primary_val_key=config.primary_val_key,
        epochs=config.epochs,
        metrics=config.metrics,
        grad_clip_val=config.grad_clip_val,
        early_stopping_patience=config.early_stopping_patience,
        experiment_name=config.experiment_name,
    )

    # Train the model
    trainer.train()


if __name__ == "__main__":
    from gnn_scheduler.configs import DEFAULT_CONFIG

    DEFAULT_CONFIG.num_chunks_train = 1
    _main(DEFAULT_CONFIG)
