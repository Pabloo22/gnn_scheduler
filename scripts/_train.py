from functools import partial
import os

import wandb
import torch
from torch_geometric.loader import DataLoader
from job_shop_lib.benchmarking import load_all_benchmark_instances

from gnn_scheduler.model import ResidualSchedulingGNN
from gnn_scheduler.data import JobShopDataset, DatasetManager
from gnn_scheduler.eval import get_performance_dataframe
from gnn_scheduler.trainer import Trainer
from gnn_scheduler.configs import Config
from gnn_scheduler.solve_jssp import solve_job_shop_with_gnn


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
    dataset_manager_train = DatasetManager(
        raw_filenames=config.train_jsons,
        dataloader_kwargs={"batch_size": config.batch_size, "shuffle": True},
    )
    val_dataset_10x10 = JobShopDataset(
        raw_filename="instances10x10_eval_0.json"
    )
    val_dataset_5x5 = JobShopDataset(raw_filename="instances5x5_eval_0.json")
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
    trainer = Trainer(
        model=model,
        train_dataloader=dataset_manager_train,
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
    best_model = trainer.model
    gnn_solver = partial(
        solve_job_shop_with_gnn,
        model=best_model,
    )
    # Evaluate the model
    performance_df = get_performance_dataframe(
        gnn_solver,
        load_all_benchmark_instances().values(),
    )
    print(performance_df)

    # Save .csv file with results
    performance_df.to_csv(
        os.path.join(trainer.checkpoint_dir, "best_model_performance.csv"),
        index=False,
    )

    # save performance metrics to wandb
    wandb.log({"performance_metrics": performance_df})

    wandb.finish()


if __name__ == "__main__":
    from gnn_scheduler.configs.experiment_configs import (
        TESTING_CONFIG,
    )

    _main(TESTING_CONFIG)
