from typing import Optional

import torch

import tqdm
from torch import optim
import wandb

from gnn_scheduler.gan import (
    set_all_seeds,
)
from gnn_scheduler.training_utils import (
    default_device,
    get_criterion_and_metric_from_task,
    Metric,
)
from gnn_scheduler.gan.model import Discriminator
from gnn_scheduler.gan.data import load_dense_data_dataset, DenseData


class RewardNetworkTrainer:

    def __init__(
        self,
        experiment_config: dict,
        use_wandb: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.experiment_config = experiment_config
        self.data_config = experiment_config["data"]
        self.model_config = experiment_config["model"]
        self.training_config = experiment_config["training"]
        self.seed = experiment_config["seed"]

        self.using_wandb = use_wandb and wandb.run is not None
        self.device = default_device() if device is None else device

        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.metric = None
        self.loss_metric = Metric(name="loss")

        if self.using_wandb:
            wandb.config.update(experiment_config)

    def compile(self):
        (
            self.train_data,
            self.val_data,
            self.test_data,
        ) = load_dense_data_dataset(**self.data_config)
        set_all_seeds(self.seed)
        self._set_model()
        self._set_metric_and_criterion()
        self._set_optimizer()

    def _set_metric_and_criterion(self):
        is_regression = self.training_config["difficulty_threshold"] is None
        self.criterion, self.metric = get_criterion_and_metric_from_task(
            is_regression=is_regression
        )

    def _set_optimizer(self):
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.training_config["learning_rate"],
            weight_decay=self.training_config["weight_decay"],
        )

    def _set_model(self):
        self.model = Discriminator(
            **self.model_config,
        )
        self.model.to(self.device)
        if self.using_wandb:
            # Watch the model to log gradients and parameters
            wandb.watch(
                self.model, log_freq=self.training_config["log_interval"]
            )

    def fit(self):
        for epoch in range(self.training_config["n_epochs"]):
            self._train_epoch(epoch)
            if self.using_wandb:
                self._log_metrics(step=epoch)

            self.evaluate(epoch=epoch)

    def _train_epoch(self, epoch: int):
        self.model.train()
        self._reset_metrics()
        for i, dense_data in tqdm.tqdm(
            enumerate(self.train_data),
            disable=not self.training_config["show_progress"],
            desc=f"Epoch {epoch}",
            total=len(self.train_data),
        ):
            self._train_step(dense_data)

            is_log_iteration = i % self.training_config["log_interval"] == 0
            if self.using_wandb and is_log_iteration and i > 0:
                self._log_metrics(
                    last_n_values=self.training_config["log_interval"]
                )

    def _reset_metrics(self):
        self.loss_metric.reset()
        self.metric.reset()

    def _train_step(self, dense_data: DenseData):
        self.optimizer.zero_grad()
        node_features, adj_matrices, label = dense_data.get_tensors(
            self.device,
            label_threshold=self.training_config["difficulty_threshold"],
        )

        output = self.model(
            node_features=node_features, adj_matrices=adj_matrices
        )
        loss = self.criterion(output, label)
        loss.backward()
        self.optimizer.step()
        self.metric.step(label.item(), output.item())
        self.loss_metric.step(loss.item())

    def _log_metrics(
        self,
        last_n_values: Optional[int] = None,
        prefix: str = "",
        step: Optional[int] = None,
    ):
        loss_name = f"{prefix}loss"
        metric_name = f"{prefix}{self.metric.name}"
        wandb.log(
            {
                loss_name: self.loss_metric.compute(last_n_values),
                metric_name: self.metric.compute(last_n_values),
            },
            step=step,
        )

    def evaluate(
        self, use_test_data: bool = False, epoch: Optional[int] = None
    ) -> tuple[float, float]:
        """Evaluates the model on the validation or test data.

        Args:
            use_test_data (bool, optional): Whether to use the test data.
                Defaults to False.

        Returns:
            tuple[float, float]: The average loss and metric.
        """
        data = self.test_data if use_test_data else self.val_data
        self._reset_metrics()
        for dense_data in tqdm.tqdm(
            data,
            disable=not self.training_config["show_progress"],
            desc="Evaluation",
        ):
            self._eval_step(dense_data)

        if self.using_wandb:
            self._log_metrics(
                prefix="test_" if use_test_data else "val_", step=epoch
            )
        return self.loss_metric.compute(), self.metric.compute()

    def _eval_step(self, dense_data: DenseData):
        node_features, adj_matrices, label = dense_data.get_tensors(
            self.device,
            label_threshold=self.training_config["difficulty_threshold"],
        )
        output = self.model(
            node_features=node_features, adj_matrices=adj_matrices
        )
        loss = self.criterion(output, label)
        self.metric.step(label.item(), output.item())
        self.loss_metric.step(loss.item())
