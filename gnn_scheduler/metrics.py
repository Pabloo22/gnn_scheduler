from abc import ABC, abstractmethod
import torch


class Metric(ABC):
    """Abstract base class for metrics.

    All metrics should implement this interface.

    Args:
        name: Optional custom name for the metric
    """

    def __init__(self, name=None):
        self.name = name or self.__class__.__name__
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        """Reset the metric state."""

    @abstractmethod
    def update(self, preds, targets) -> None:
        """Update the metric with a new batch of predictions and targets.

        Args:
            preds: Predictions from the model
            targets: Ground truth targets
        """

    @abstractmethod
    def compute(self) -> float:
        """Compute the metric value.

        Returns:
            The metric value
        """

    def __str__(self):
        return self.name


class Accuracy(Metric):
    """Accuracy metric for binary classification with BCEWithLogitsLoss.

    This metric applies sigmoid to the model outputs and uses a threshold
    (default: 0.5) to convert logits to binary predictions. It handles
    variable-sized batches by accumulating counts properly.

    Args:
        threshold: Threshold to convert probabilities to binary predictions
        name: Name of the metric
    """

    def __init__(self, threshold=0.5, name="Accuracy"):
        self.threshold = threshold
        super().__init__(name=name)

    def reset(self):
        """Reset the metric state."""
        self.correct = 0
        self.total = 0

    def update(self, preds, targets):
        """Update the metric with a new batch of predictions and targets.

        Args:
            preds: Logits from the model (before sigmoid)
            targets: Ground truth binary targets
        """
        with torch.no_grad():
            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(preds)

            # Convert probabilities to binary predictions
            binary_preds = (probs > self.threshold).float()

            # Count correct predictions
            correct = (binary_preds == targets).float().sum().item()

            # Update totals
            self.correct += correct
            self.total += targets.numel()

    def compute(self):
        """Compute the accuracy.

        Returns:
            Accuracy as a float between 0 and 1
        """
        if self.total == 0:
            return 0.0
        return self.correct / self.total


class Precision(Metric):
    """Precision metric for binary classification with BCEWithLogitsLoss.

    Precision = True Positives / (True Positives + False Positives)

    Args:
        threshold: Threshold to convert probabilities to binary predictions
        name: Name of the metric
    """

    def __init__(self, threshold=0.5, name="Precision"):
        self.threshold = threshold
        super().__init__(name=name)

    def reset(self):
        """Reset the metric state."""
        self.true_positives = 0
        self.predicted_positives = 0

    def update(self, preds, targets):
        """Update the metric with a new batch of predictions and targets."""
        with torch.no_grad():
            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(preds)

            # Convert probabilities to binary predictions
            binary_preds = (probs > self.threshold).float()

            # Count true positives and predicted positives
            self.true_positives += (binary_preds * targets).sum().item()
            self.predicted_positives += binary_preds.sum().item()

    def compute(self):
        """Compute precision."""
        if self.predicted_positives == 0:
            return 0.0
        return self.true_positives / self.predicted_positives


class Recall(Metric):
    """Recall metric for binary classification with BCEWithLogitsLoss.

    Recall = True Positives / (True Positives + False Negatives)

    Args:
        threshold: Threshold to convert probabilities to binary predictions
        name: Name of the metric
    """

    def __init__(self, threshold=0.5, name="Recall"):
        self.threshold = threshold
        super().__init__(name=name)

    def reset(self):
        """Reset the metric state."""
        self.true_positives = 0
        self.actual_positives = 0

    def update(self, preds, targets):
        """Update the metric with a new batch of predictions and targets."""
        with torch.no_grad():
            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(preds)

            # Convert probabilities to binary predictions
            binary_preds = (probs > self.threshold).float()

            # Count true positives and actual positives
            self.true_positives += (binary_preds * targets).sum().item()
            self.actual_positives += targets.sum().item()

    def compute(self):
        """Compute recall."""
        if self.actual_positives == 0:
            return 0.0
        return self.true_positives / self.actual_positives


class F1Score(Metric):
    """F1 Score metric for binary classification with BCEWithLogitsLoss.

    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    Args:
        threshold: Threshold to convert probabilities to binary predictions
        name: Name of the metric
    """

    def __init__(self, threshold=0.5, name="F1Score"):
        self.threshold = threshold
        self.precision = Precision(threshold)
        self.recall = Recall(threshold)
        super().__init__(name=name)

    def reset(self):
        """Reset the metric state."""
        self.precision.reset()
        self.recall.reset()

    def update(self, preds, targets):
        """Update the metric with a new batch of predictions and targets."""
        self.precision.update(preds, targets)
        self.recall.update(preds, targets)

    def compute(self):
        """Compute F1 score."""
        precision = self.precision.compute()
        recall = self.recall.compute()

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)
