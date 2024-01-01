"""Contains functions that create PyTorch objects based on a string 
identifier."""
import torch

from gnn_scheduler.training_utils import Accuracy, MeanAbsoluteError, Metric


def get_activation_function(name: str | None) -> torch.nn.Module:
    """Returns a PyTorch activation function based on a string identifier.

    Args:
        name (str): String identifier of the activation function.

    Raises:
        ValueError: If the string identifier is not supported.

    Returns:
        torch.nn.Module: PyTorch activation function.
    """
    activation_functions = {
        "relu": torch.nn.ReLU,
        "leaky_relu": torch.nn.LeakyReLU,
        "elu": torch.nn.ELU,
        "sigmoid": torch.nn.Sigmoid,
        "tanh": torch.nn.Tanh,
    }

    if name is None:
        return torch.nn.Identity()

    if name not in activation_functions:
        raise ValueError(
            f"Activation function {name} not supported. Supported "
            f"activation functions are: {list(activation_functions.keys())}"
        )

    return activation_functions[name]()


def get_loss_function_from_task(
    is_regression: bool, is_binary: bool = True, with_logits: bool = True
) -> torch.nn.Module:
    """Gets the appropriate loss function based on task type and parameters.

    Args:
        is_regression (bool): If True, the task is regression, otherwise it's
            classification.
        output_dim (int): The dimension of the output. 1 for binary
            classification, >1 for multi-class.
        with_logits (bool): If True, use the version of the loss function
            that expects logits.

    Returns:
    torch.nn.Module: The appropriate loss function.
    """
    if is_regression:
        return torch.nn.MSELoss()

    if is_binary:
        return (
            torch.nn.BCEWithLogitsLoss() if with_logits else torch.nn.BCELoss()
        )

    return torch.nn.CrossEntropyLoss() if with_logits else torch.nn.NLLLoss()


def get_metric_from_task(
    is_regression: bool, accuracy_threshold: float = 0.5
) -> Metric:
    """Returns the metric for the given task."""
    return (
        MeanAbsoluteError()
        if is_regression
        else Accuracy(threshold=accuracy_threshold)
    )


def get_criterion_and_metric_from_task(
    is_regression: bool,
    is_binary: bool = True,
    accuracy_threshold: float = 0.5,
    with_logits: bool = True,
) -> tuple[torch.nn.Module, Metric]:
    """Returns the criterion and metric for the given task."""
    criterion = get_loss_function_from_task(
        is_regression=is_regression,
        is_binary=is_binary,
        with_logits=with_logits,
    )
    metric = get_metric_from_task(
        is_regression=is_regression, accuracy_threshold=accuracy_threshold
    )

    return criterion, metric


def default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
