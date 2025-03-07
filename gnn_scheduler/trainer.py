import os
from typing import Optional, Any, TypedDict
import torch
import wandb
from tqdm import tqdm

from torch_geometric.loader import DataLoader

from gnn_scheduler.data import JobShopData
from gnn_scheduler.metrics import Metric


class ResultDict(TypedDict):
    loss: float
    metrics: dict[str, float]


class Trainer:
    """A PyTorch trainer class that handles training, validation, model
    checkpointing, and tracking metrics with Weights & Biases.

    Args:
        model:
            PyTorch model to train
        train_dataloader:
            DataLoader for training data
        val_dataloaders:
            Dictionary of validation DataLoaders (name -> dataloader)
        primary_val_key:
            Key in val_dataloaders to use for best model selection
        criterion:
            Loss function
        optimizer:
            Optimizer for model parameters
        scheduler:
            Learning rate scheduler (optional)
        epochs:
            Number of training epochs
        device:
            Device to use for training (defaults to CUDA if available)
        checkpoint_dir:
            Directory to save model checkpoints
        metric_mode:
            'max' if higher metric is better, 'min' if lower is better
        metrics:
            List of Metric objects to compute during training/validation
        primary_metric:
            Name of the metric to use for best model selection
        grad_clip_val:
            Value for gradient clipping (optional)
        early_stopping_patience:
            Number of epochs to wait before early stopping (optional)
        experiment_name:
            Name for W&B experiment (optional)
        project_name:
            Name for W&B project
        wandb_config:
            Additional config parameters to log to W&B
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        val_dataloaders: dict[str, DataLoader],
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
        primary_val_key: str = "instances10x10_val",
        grad_clip_val: Optional[float] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        epochs: int = 10,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "checkpoints",
        metric_mode: str = "max",
        metrics: Optional[list[Metric]] = None,
        primary_metric: Optional[str] = "Accuracy",
        early_stopping_patience: Optional[int] = None,
        experiment_name: Optional[str] = None,
        project_name: str = "job-shop-imitation-learning",
        wandb_config: Optional[dict[str, Any]] = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloaders = val_dataloaders

        if primary_val_key not in val_dataloaders:
            raise ValueError(
                f"Primary validation key '{primary_val_key}' not found in "
                "val_dataloaders"
            )
        self.primary_val_key = primary_val_key

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs

        # Set device if not provided
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        self.model = self.model.to(self.device)

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        if metric_mode not in ["max", "min"]:
            raise ValueError("metric_mode must be either 'max' or 'min'")
        self.metric_mode = metric_mode

        self.grad_clip_val = grad_clip_val

        # Initialize metrics
        self.metrics = metrics or []

        # Determine primary metric
        if primary_metric is not None:
            # Check if primary_metric exists in metrics
            metric_names = [metric.name for metric in self.metrics]
            if primary_metric not in metric_names:
                raise ValueError(
                    f"Primary metric '{primary_metric}' not found in provided "
                    "metrics"
                )
            self.primary_metric: str | None = primary_metric
        elif self.metrics:
            # Use the first metric as primary if there are metrics
            self.primary_metric = self.metrics[0].name
        else:
            # Fall back to loss if no metrics
            self.primary_metric = None

        self.early_stopping_patience = early_stopping_patience

        # Weights & Biases setup
        self.experiment_name = (
            experiment_name or f"train_{self.model.__class__.__name__}"
        )
        self.project_name = project_name

        # Metrics tracking
        self.best_metric_value = (
            float("-inf") if self.metric_mode == "max" else float("inf")
        )
        self.best_epoch = 0
        self.epochs_without_improvement = 0

        # Initialize W&B
        config = wandb_config or {}
        config.update(
            {
                "model": self.model.__class__.__name__,
                "optimizer": self.optimizer.__class__.__name__,
                "scheduler": (
                    self.scheduler.__class__.__name__
                    if self.scheduler
                    else None
                ),
                "criterion": self.criterion.__class__.__name__,
                "epochs": self.epochs,
                "batch_size": getattr(
                    self.train_dataloader, "batch_size", None
                ),
                "device": str(self.device),
                "grad_clip_val": self.grad_clip_val,
                "metrics": [metric.name for metric in self.metrics],
                "primary_metric": self.primary_metric,
            }
        )
        wandb.init(
            project=self.project_name, name=self.experiment_name, config=config
        )

        # Log model architecture
        wandb.watch(self.model, self.criterion, log="all", log_freq=10)

    def _reset_metrics(self):
        """Reset all metrics."""
        for metric in self.metrics:
            metric.reset()

    def _update_metrics(self, outputs, targets):
        """Update all metrics with the current batch.

        Args:
            outputs: Model outputs (predictions)
            targets: Ground truth targets
        """
        for metric in self.metrics:
            metric.update(outputs, targets)

    def _compute_metrics(self) -> dict[str, float]:
        """Compute values for all metrics.

        Returns:
            Dictionary mapping metric names to values
        """
        return {metric.name: metric.compute() for metric in self.metrics}

    def train(self) -> dict[str, list[float]]:
        """Full training process for the specified number of epochs.

        Returns:
            Dictionary containing training history
        """
        history: dict[str, list[float]] = {
            "train_loss": [],
            "learning_rate": [],
        }

        # Initialize metrics in history
        for metric in self.metrics:
            history[f"train_{metric.name}"] = []

        # Initialize validation metrics in history
        for val_key in self.val_dataloaders.keys():
            history[f"{val_key}_loss"] = []
            for metric in self.metrics:
                history[f"{val_key}_{metric.name}"] = []

        print(f"Starting training on {self.device}")

        try:
            for epoch in range(1, self.epochs + 1):
                print(f"\nEpoch {epoch}/{self.epochs}")

                # Training phase
                train_loss, train_metrics = self._train_epoch()
                history["train_loss"].append(train_loss)

                # Record training metrics
                for metric_name, metric_value in train_metrics.items():
                    history[f"train_{metric_name}"].append(metric_value)

                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]["lr"]
                history["learning_rate"].append(current_lr)

                # Validation phase
                val_results: dict[str, ResultDict] = {}
                for val_key, val_dataloader in self.val_dataloaders.items():
                    val_loss, val_metrics = self._validate_epoch(
                        val_dataloader
                    )
                    val_results[val_key] = {
                        "loss": val_loss,
                        "metrics": val_metrics,
                    }

                    history[f"{val_key}_loss"].append(val_loss)
                    for metric_name, metric_value in val_metrics.items():
                        history[f"{val_key}_{metric_name}"].append(
                            metric_value
                        )

                # Update learning rate scheduler if exists
                if self.scheduler is not None:
                    if isinstance(
                        self.scheduler,
                        torch.optim.lr_scheduler.ReduceLROnPlateau,
                    ):
                        # Use primary metric for scheduler if available
                        primary_val_dict = val_results[self.primary_val_key]
                        if self.primary_metric:
                            primary_val_metrics = primary_val_dict["metrics"]
                            primary_val = primary_val_metrics[
                                self.primary_metric
                            ]
                            self.scheduler.step(primary_val)
                        else:
                            # Fall back to loss
                            primary_loss = primary_val_dict["loss"]
                            self.scheduler.step(primary_loss)
                    else:
                        self.scheduler.step()

                # Log metrics to W&B
                metrics_to_log = {
                    "train/loss": train_loss,
                    "train/learning_rate": current_lr,
                }

                # Log training metrics
                for metric_name, metric_value in train_metrics.items():
                    metrics_to_log[f"train/{metric_name}"] = metric_value

                # Log validation metrics
                for val_key, results in val_results.items():
                    metrics_to_log[f"val/{val_key}/loss"] = results["loss"]
                    for metric_name, metric_value in results[
                        "metrics"
                    ].items():
                        metrics_to_log[f"val/{val_key}/{metric_name}"] = (
                            metric_value
                        )

                wandb.log(metrics_to_log, step=epoch)

                # Check if this is the best model on primary validation set
                is_best = False
                if self.primary_metric:
                    # Use the primary metric if available
                    primary_val = val_results[self.primary_val_key]["metrics"][
                        self.primary_metric
                    ]
                    is_best = self._is_better_metric(primary_val)

                    if is_best:
                        print(
                            f"New best model with {self.primary_val_key} "
                            f"{self.primary_metric}: {primary_val:.6f}"
                        )
                        self.best_metric_value = primary_val
                        self.best_epoch = epoch
                        self.epochs_without_improvement = 0
                        self._save_checkpoint(epoch, primary_val, is_best=True)
                    else:
                        self.epochs_without_improvement += 1
                        self._save_checkpoint(
                            epoch, primary_val, is_best=False
                        )
                else:
                    # Fall back to loss if no primary metric
                    val_loss = val_results[self.primary_val_key]["loss"]
                    # For loss, better means lower (min mode)
                    is_best = self._is_better_metric(
                        -val_loss if self.metric_mode == "max" else val_loss
                    )

                    if is_best:
                        print(
                            f"New best model with {self.primary_val_key} "
                            f"loss: {val_loss:.6f}"
                        )
                        self.best_metric_value = val_loss
                        self.best_epoch = epoch
                        self.epochs_without_improvement = 0
                        self._save_checkpoint(epoch, val_loss, is_best=True)
                    else:
                        self.epochs_without_improvement += 1
                        self._save_checkpoint(epoch, val_loss, is_best=False)

                # Print epoch summary
                self._print_epoch_summary(
                    epoch, train_loss, train_metrics, val_results
                )

                # Check for early stopping
                if self._should_early_stop():
                    print(f"Early stopping triggered after {epoch} epochs")
                    break

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")

        finally:
            # Finish the W&B run
            wandb.finish()

            # Load the best model
            self._load_best_model()

            if self.primary_metric:
                print(
                    "\nTraining completed. Best model from epoch "
                    f"{self.best_epoch} with {self.primary_val_key} "
                    f"{self.primary_metric}: {self.best_metric_value:.6f}"
                )
            else:
                print(
                    f"\nTraining completed. Best model from epoch "
                    f"{self.best_epoch} with {self.primary_val_key} "
                    f"loss: {self.best_metric_value:.6f}"
                )

        return history

    def _train_epoch(self) -> tuple[float, dict[str, float]]:
        """
        Train the model for a single epoch.

        Returns:
            Tuple of (average loss, metrics dictionary)
        """
        self.model.train()
        epoch_loss = 0.0

        # Reset metrics
        self._reset_metrics()

        pbar = tqdm(self.train_dataloader, desc="Training")
        for batch in pbar:
            # Move batch to device
            inputs, targets = self._prepare_batch(batch)
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            loss.backward()

            # Gradient clipping if enabled
            if self.grad_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_val
                )

            # Update weights
            self.optimizer.step()

            # Update metrics
            self._update_metrics(outputs, targets)

            # Update statistics
            batch_loss = loss.item()
            epoch_loss += batch_loss

            # Get current metrics for progress bar
            batch_metrics = self._compute_metrics()

            # Update progress bar with all metrics
            if self.metrics:
                postfix = {"loss": f"{batch_loss:.6f}"}
                # Add all metrics to the progress bar
                for metric in self.metrics:
                    metric_name = metric.name
                    metric_val = batch_metrics[metric_name]
                    postfix[metric_name] = f"{metric_val:.4f}"
                pbar.set_postfix(postfix)
            else:
                pbar.set_postfix({"loss": f"{batch_loss:.6f}"})

        # Calculate average loss for the epoch
        avg_loss = epoch_loss / len(self.train_dataloader)

        # Compute final metrics
        metrics = self._compute_metrics()

        return avg_loss, metrics

    def _print_epoch_summary(
        self,
        epoch: int,
        train_loss: float,
        train_metrics: dict[str, float],
        val_results: dict[str, ResultDict],
    ) -> None:
        """Prints a summary of the epoch results.

        Args:
            epoch: Current epoch number
            train_loss: Training loss
            train_metrics: Dictionary of training metrics
            val_results: Dictionary of validation results. Each key is a
                validation set name and the value is a dictionary with keys
                "loss" and "metrics" containing the loss and metric
                values respectively.
        """
        # Create training summary
        train_summary = [
            f"Epoch {epoch}/{self.epochs} - train_loss: {train_loss:.6f}"
        ]

        # Add training metrics
        for metric_name, metric_value in train_metrics.items():
            train_summary.append(f"train_{metric_name}: {metric_value:.6f}")

        print(" | ".join(train_summary))

        # Create validation summary
        for val_key, results in val_results.items():
            val_loss = results["loss"]
            val_metrics = results["metrics"]

            val_summary = [f"{val_key}_loss: {val_loss:.6f}"]

            # Add validation metrics
            for metric_name, metric_value in val_metrics.items():
                if (
                    val_key == self.primary_val_key
                    and metric_name == self.primary_metric
                ):
                    val_summary.append(
                        f"{val_key}_{metric_name}: {metric_value:.6f} "
                        "(primary)"
                    )
                else:
                    val_summary.append(
                        f"{val_key}_{metric_name}: {metric_value:.6f}"
                    )

            print(" | ".join(val_summary))

    def _validate_epoch(
        self, dataloader: DataLoader
    ) -> tuple[float, dict[str, float]]:
        """
        Validate the model on a dataloader.

        Args:
            dataloader: Validation dataloader

        Returns:
            Tuple of (average loss, metrics dictionary)
        """
        self.model.eval()
        epoch_loss = 0.0

        # Reset metrics
        self._reset_metrics()

        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Validating")
            for batch in pbar:
                # Move batch to device
                inputs, targets = self._prepare_batch(batch)

                # Forward pass
                outputs = self.model(inputs)

                # Calculate loss
                loss = self.criterion(outputs, targets)

                # Update metrics
                self._update_metrics(outputs, targets)

                # Update statistics
                batch_loss = loss.item()
                epoch_loss += batch_loss

                # Get current metrics for progress baºr
                batch_metrics = self._compute_metrics()

                # Update progress bar with first metric
                if self.metrics:
                    first_metric = self.metrics[0]
                    metric_val = batch_metrics[first_metric.name]
                    pbar.set_postfix(
                        {
                            "loss": f"{batch_loss:.6f}",
                            f"{first_metric.name}": f"{metric_val:.4f}",
                        }
                    )
                else:
                    pbar.set_postfix({"loss": f"{batch_loss:.6f}"})

        # Calculate average loss
        avg_loss = epoch_loss / len(dataloader)

        # Compute final metrics
        metrics = self._compute_metrics()

        return avg_loss, metrics

    def _prepare_batch(self, batch: JobShopData) -> tuple:
        """Prepare a batch for training/validation by moving it to the device.

        Args:
            batch: Input batch which can be in various formats

        Returns:
            Tuple of (inputs, targets) moved to device
        """
        if isinstance(batch, JobShopData):
            inputs = batch.to(self.device)
            targets = batch.y.to(self.device)
            return inputs, targets

        raise ValueError(f"Unsupported batch type: {type(batch)}")

    def _is_better_metric(self, metric_value: float) -> bool:
        """
        Check if the current metric is better than the best so far.

        Args:
            metric_value: Current metric value

        Returns:
            True if current metric is better than best
        """
        if self.metric_mode == "max":
            return metric_value > self.best_metric_value

        return metric_value < self.best_metric_value

    def _save_checkpoint(
        self, epoch: int, metric: float, is_best: bool = False
    ) -> None:
        """Saves a model checkpoint.

        Args:
            epoch: Current epoch number
            metric: Validation metric value
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metric": metric,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"
        )
        torch.save(checkpoint, checkpoint_path)

        # Save best model if applicable
        if not is_best:
            return

        best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        torch.save(checkpoint, best_path)

        # Log best model to W&B
        assert wandb.run is not None
        wandb.run.summary["best_epoch"] = epoch
        wandb.run.summary["best_metric"] = metric

        # Save model to W&B artifacts
        artifact = wandb.Artifact(
            name=f"{self.experiment_name}_best_model", type="model"
        )
        artifact.add_file(best_path)
        wandb.log_artifact(artifact)

    def _load_best_model(self) -> None:
        """Load the best model from checkpoint."""
        best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded best model from epoch {checkpoint['epoch']}")

    def _should_early_stop(self) -> bool:
        """
        Check if training should be stopped early.

        Returns:
            True if early stopping criteria is met
        """
        if self.early_stopping_patience is None:
            return False

        return self.epochs_without_improvement >= self.early_stopping_patience

    def predict(
        self, dataloader: DataLoader
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], dict[str, float]]:
        """
        Generate predictions for a dataloader.

        Args:
            dataloader: DataLoader for prediction

        Returns:
            Tuple of (all_outputs, all_targets, metrics_dict)
        """
        self.model.eval()
        all_outputs = []
        all_targets = []

        # Reset metrics
        self._reset_metrics()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                inputs, targets = self._prepare_batch(batch)
                outputs = self.model(inputs)

                all_outputs.append(outputs.cpu())
                if targets is None:
                    continue
                all_targets.append(targets.cpu())
                # Update metrics
                self._update_metrics(outputs, targets)

        # Concatenate all outputs and targets
        all_outputs_tensor = torch.cat(all_outputs, dim=0)

        # Compute metrics
        metrics = self._compute_metrics() if all_targets else {}

        if all_targets:
            all_targets_tensor = torch.cat(all_targets, dim=0)
            return all_outputs_tensor, all_targets_tensor, metrics

        return all_outputs_tensor, None, metrics
