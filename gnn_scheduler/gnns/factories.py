"""Contains functions that create PyTorch objects based on a string 
identifier."""
import torch


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
