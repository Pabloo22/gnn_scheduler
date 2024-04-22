from typing import TypeVar

from sklearn.model_selection import train_test_split


_T = TypeVar("_T")


def train_eval_test_split(
    instances: list[_T],
    seed: int = 0,
    eval_size: float = 0.1,
    test_size: float = 0.2,
) -> tuple[list[_T], list[_T], list[_T]]:
    """Returns the train, eval and test sets of instances.

    The train set is further divided into train and eval sets using the
    train_test_split function from scikit-learn.

    Args:
        instances (list[_T]): the instances to split
        seed (int, optional): the seed for the train test split. Defaults to 0.
        eval_size (float, optional): the proportion of instances to use for
            evaluation within the train set. Defaults to 0.1.
        test_size (float, optional): the proportion of instances to use for
            testing. Defaults to 0.2.

    Returns:
        tuple[list[_T], list[_T], list[_T]]: the train, eval and test sets.
    """
    train, test = train_test_split(
        instances, test_size=test_size, random_state=seed
    )
    train, evaluation = train_test_split(
        train, test_size=eval_size, random_state=seed
    )

    return train, evaluation, test
