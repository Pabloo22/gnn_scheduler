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
    # train_dataset = JobShopDataset(
    #     num_chunks=config.num_chunks_train,
    #     max_chunks_in_memory=config.max_chunks_in_memory,
    #     processed_filenames_prefix="instances_train10x10",
    #     raw_filenames=config.train_jsons,
    # )
    dataset_manager_train = JobShopDataset(
        raw_filenames=config.train_jsons,
        processed_filenames_prefix=config.processed_filenames_prefix_train,
        num_chunks=config.num_chunks_train,
        max_chunks_in_memory=config.max_chunks_in_memory,
    )
    val_dataset_10x10 = JobShopDataset(
        num_chunks=1,
        raw_filenames="instances10x10_eval_0.json",
        processed_filenames_prefix="instances10x10_eval",
    )
    val_dataset_5x5 = JobShopDataset(
        num_chunks=1,
        raw_filenames="instances5x5_eval_0.json",
        processed_filenames_prefix="instances5x5_eval",
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
    for _ in range(config.epochs):
        for train_dataset in dataset_manager_train:
            train_dataloader = DataLoader(
                train_dataset, batch_size=config.batch_size, shuffle=True
            )

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
                epochs=1,
                metrics=config.metrics,
                grad_clip_val=config.grad_clip_val,
                early_stopping_patience=config.early_stopping_patience,
                experiment_name=config.experiment_name,
            )

            # Train the model
            trainer.train()


if __name__ == "__main__":
    from gnn_scheduler.configs.experiment_configs import (
        EXPERIMENT_3,
    )

    _main(EXPERIMENT_3)
