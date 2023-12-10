import os
from typing import TypeVar
from omegaconf import OmegaConf

_T = TypeVar("_T")


def load_yaml(path: os.PathLike | str | bytes, structure: _T) -> _T:
    """Loads a yaml file and returns a structure object.

    Args:
        path (os.PathLike | str | bytes): Path to the yaml file.
        structure (_T): A dataclass that represents the structure of the
            yaml file.

    Returns:
        OmegaConf: OmegaConf object.
    """
    config_dict = OmegaConf.load(path)
    return structure(**config_dict)
