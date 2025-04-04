from functools import partial
import torch
from torch_geometric.loader import (  # type: ignore[import-untyped]
    DataLoader,
    NeighborLoader,
)
from job_shop_lib.benchmarking import load_all_benchmark_instances
import wandb
from gnn_scheduler.model import ResidualSchedulingGNN
from gnn_scheduler.data import (
    JobShopDataset,
    DatasetManager,
    CombinedJobShopDataset,
)
from gnn_scheduler.eval import get_performance_dataframe, load_model
from gnn_scheduler.trainer import Trainer
from gnn_scheduler.configs import Config
from gnn_scheduler.solve_jssp import solve_job_shop_with_gnn
from gnn_scheduler.utils import get_data_path, get_project_path


def _run_experiment(config: Config):
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
    dataset_manager_train: DatasetManager | DataLoader = DatasetManager(
        raw_filenames=config.train_jsons,
        dataloader_kwargs={"batch_size": config.batch_size, "shuffle": True},
        dataset_kwargs={"store_each_n_steps": config.store_each_n_steps},
    )
    if config.use_combined_dataset:
        combined_dataset = CombinedJobShopDataset(
            dataset_manager_train,
            processed_filename=config.combined_dataset_filename,
        )
        if config.neighbor_sampling is None:
            dataset_manager_train = DataLoader(
                combined_dataset, batch_size=config.batch_size, shuffle=True
            )
        else:
            dataset_manager_train = NeighborLoader(
                combined_dataset,
                num_neighbors=[config.neighbor_sampling]
                * config.model_config.num_layers,
                batch_size=config.batch_size,
                shuffle=True,
            )

    val_dataset = JobShopDataset(raw_filename=config.val_dataset_filename)
    if config.neighbor_sampling is None:
        val_dataloader = DataLoader(
            val_dataset, batch_size=config.batch_size
        )
    else:
        val_dataloader = NeighborLoader(
            val_dataset,
            num_neighbors=[config.neighbor_sampling]
            * config.model_config.num_layers,
            batch_size=config.batch_size,
            shuffle=True,
        )
    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    # Create loss function
    criterion = torch.nn.BCEWithLogitsLoss()
    trainer = Trainer(
        model=model,
        train_dataloader=dataset_manager_train,
        val_dataloaders={
            config.primary_val_key: val_dataloader,
        },
        optimizer=optimizer,
        criterion=criterion,
        primary_val_key=config.primary_val_key,
        epochs=config.epochs,
        metrics=config.metrics,
        grad_clip_val=config.grad_clip_val,
        early_stopping_patience=config.early_stopping_patience,
        experiment_name=config.experiment_name,
        n_batches_per_epoch=config.n_batches_per_epoch,
        eval_instances=config.eval_instances,
    )

    # Train the model
    trainer.train()
    best_model = trainer.model
    gnn_solver = partial(
        solve_job_shop_with_gnn,
        model=best_model,  # type: ignore[arg-type]
        allow_operation_reservation=config.allow_operation_reservation,
        neighborhood_sampling=config.neighbor_sampling,
    )
    # Evaluate the model
    performance_df = get_performance_dataframe(
        gnn_solver,
        load_all_benchmark_instances().values(),
    )
    print(performance_df)

    # Save .csv file with results
    performance_df.to_csv(
        get_data_path() / f"{config.experiment_name}_results.csv",
        index=False,
    )

    # save performance metrics to wandb
    wandb.log({"performance_metrics": performance_df})

    # Save performance_df to csv
    performance_df.to_csv(
        get_data_path() / f"{config.experiment_name}_results.csv",
        index=False,
    )

    # Aggregate by problem size (num_jobs and num_machines columns)
    performance_df_agg = (
        performance_df[["num_jobs", "num_machines", "optimality_gap"]]
        .groupby(["num_jobs", "num_machines"])
        .mean()
    ).reset_index()
    wandb.log({"performance_metrics_agg": performance_df_agg})

    wandb.finish()


def resume_training(
    config: Config,
    checkpoint_path: str | None = None,
    best_val_metric: float | None = None,
):
    """
    Resume training from a saved checkpoint.

    Args:
        config (Config): Configuration object for the experiment
        checkpoint_path (str, optional): Path to the checkpoint file.
                                         If None, will use the default path based on experiment name.
    """

    # Set up wandb for continued logging
    wandb_run_name = f"{config.experiment_name}_resumed"
    wandb.init(project="job-shop-imitation-learning", name=wandb_run_name)

    if checkpoint_path is None:
        checkpoint_path = (
            get_project_path()
            / "checkpoints"
            / config.experiment_name
            / "best_model.pth"
        )
    checkpoint = torch.load(checkpoint_path)
    best_model = load_model(checkpoint_path, config)
    best_model.train()
    dataset_manager_train: DatasetManager | DataLoader = DatasetManager(
        raw_filenames=config.train_jsons,
        dataloader_kwargs={"batch_size": config.batch_size, "shuffle": True},
        dataset_kwargs={"store_each_n_steps": config.store_each_n_steps},
    )

    if config.use_combined_dataset:
        combined_dataset = CombinedJobShopDataset(
            dataset_manager_train,
            processed_filename=config.combined_dataset_filename,
        )
        dataset_manager_train = DataLoader(
            combined_dataset, batch_size=config.batch_size, shuffle=True
        )

    val_dataset = JobShopDataset(raw_filename=config.val_dataset_filename)
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.batch_size
    )

    # Set up optimizer with the same parameters
    optimizer = torch.optim.AdamW(best_model.parameters(), lr=config.lr)

    # Load optimizer state if available
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        print(
            "No optimizer state found in checkpoint. Using default"
            "optimizer."
        )

    # Create loss function
    criterion = torch.nn.BCEWithLogitsLoss()

    # Set up the trainer with the loaded model
    trainer = Trainer(
        model=best_model,
        train_dataloader=dataset_manager_train,
        val_dataloaders={
            config.primary_val_key: val_dataloader,
        },
        optimizer=optimizer,
        criterion=criterion,
        primary_val_key=config.primary_val_key,
        epochs=config.epochs,  # This will be the total epochs to train
        metrics=config.metrics,
        grad_clip_val=config.grad_clip_val,
        early_stopping_patience=config.early_stopping_patience,
        experiment_name=wandb_run_name,
        n_batches_per_epoch=config.n_batches_per_epoch,
        eval_instances=config.eval_instances,
        allow_operation_reservation=config.allow_operation_reservation,
    )
    if best_val_metric is None:
        raise ValueError(
            "best_val_metric must be provided to resume training."
        )
    trainer.best_metric_value = best_val_metric
    # Train the model (continuing from where it left off)
    trainer.train()

    # Evaluate the model after resumed training
    best_model = trainer.model  # type: ignore[assignment]
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

    # Save results to CSV
    performance_df.to_csv(
        get_data_path() / f"{wandb_run_name}_results.csv",
        index=False,
    )

    # Log performance metrics to wandb
    wandb.log({"performance_metrics": performance_df})

    # Aggregate and log performance metrics by problem size
    performance_df_agg = (
        performance_df[["num_jobs", "num_machines", "optimality_gap"]]
        .groupby(["num_jobs", "num_machines"])
        .mean()
    ).reset_index()
    wandb.log({"performance_metrics_agg": performance_df_agg})

    wandb.finish()


def evaluate_model_in_crashed_run(config: Config):
    name = config.experiment_name + "_evaluated"
    wandb.init(project="job-shop-imitation-learning", name=name)
    model_path = str(
        get_project_path()
        / "checkpoints"
        / config.experiment_name
        / "best_model.pth"
    )
    best_model = load_model(model_path, config)

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
        get_data_path() / f"{config.experiment_name}_results.csv",
        index=False,
    )

    # save performance metrics to wandb
    wandb.log({"performance_metrics": performance_df})

    # Save performance_df to csv
    performance_df.to_csv(
        get_data_path() / f"{config.experiment_name}_results.csv",
        index=False,
    )
    wandb.finish()


def _main(config: Config):
    num_runs = config.num_runs
    original_experiment_name = config.experiment_name
    for run in range(1, num_runs + 1):
        if num_runs > 1:
            config.experiment_name = f"{original_experiment_name}_run_{run}"
        _run_experiment(config)


if __name__ == "__main__":
    from gnn_scheduler.configs.experiment_configs import *

    _main(EXPERIMENT_39)
    _main(EXPERIMENT_40)
    _main(EXPERIMENT_41)
    _main(EXPERIMENT_42)
