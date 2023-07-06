"""Contains the abstract class for the games."""
from __future__ import annotations

import abc
from copy import deepcopy
from typing import TypeVar, Optional

import gymnasium as gym
import numpy as np

State = TypeVar("State")


class Game(gym.Env):
    """Abstract class for a game."""

    def __init__(self, initial_state: Optional[State] = None):
        self.initial_state = initial_state
        self.current_state = deepcopy(initial_state)
        self.current_player = 1

    @staticmethod
    @abc.abstractmethod
    def legal_moves(state: State) -> list[int] | np.ndarray:
        """Returns the legal moves for the current player."""

    @staticmethod
    @abc.abstractmethod
    def next_state(state: State, action: int) -> State:
        """Returns the next state given the current state and action."""

    @staticmethod
    @abc.abstractmethod
    def get_reward(state: State) -> int:
        """Returns the reward for the current state."""

    @staticmethod
    @abc.abstractmethod
    def is_finished(state: State) -> bool:
        """Returns True if the game is finished."""
