from typing import Any, Optional


class Metric:
    def __init__(self, name: str) -> None:
        self.name = name
        self.history = []

    def step(self, *args):
        value = self(*args)
        self.history.append(value)

    def reset(self):
        self.history = []

    def compute(self, last_n_values: Optional[int] = None):
        """Computes the metric for the last n values.

        Args:
            last_n_values (Optional[int], optional): The number of values to
                consider. Defaults to None, which means all values.
        """
        last_n_values = (
            len(self.history) if last_n_values is None else last_n_values
        )
        return sum(self.history[-last_n_values:]) / last_n_values

    def __call__(self, *args: Any):
        return args[0]


class Accuracy(Metric):
    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__(name="accuracy")
        self.success_count = 0
        self.total_count = 0
        self.threshold = threshold

    def __call__(self, y_true, y_pred):
        true_positive = y_pred >= self.threshold and y_true >= self.threshold
        true_negative = y_pred < self.threshold and y_true < self.threshold
        return int(true_positive or true_negative)


class MeanAbsoluteError(Metric):
    def __init__(self) -> None:
        super().__init__(name="mae")

    def __call__(self, y_true, y_pred):
        return abs(y_true - y_pred)
