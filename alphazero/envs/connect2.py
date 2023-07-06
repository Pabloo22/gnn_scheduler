"""Contains the Connect2 class"""
from typing import Optional, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from alphazero.base import Game

Connect2State = np.typing.NDArray[np.int8]
GRID_SIZE = 4


class Connect2(Game):
    """Connect2 environment.

    The environment consists of a 1x4 grid. Each player takes turns placing
    their chip on the grid. The first player to have two consecutive chips
    wins.

    The observation space is a 1x4 grid with 0 representing an empty space,
    1 representing a chip from player 1, and -1 representing a chip from
    player 2.

    The action space is a 1x4 grid with 0 representing the leftmost space
    and 3 representing the rightmost space.

    The reward is 0 if the game is not over, 1 if player 1 wins, and -1 if
    player 2 (encoded as player -1) wins.

    The game is over if the board is full or if a player has two consecutive
    chips.

    Attributes:
        metadata: The metadata for the environment.
        size: The size of the grid (1 x size). Defaults to 4.
        observation_space: The observation space.
        action_space: The action space.
        initial_state: The initial state of the environment.
        current_state: The current state of the environment.
        current_player: The current player.
        render_mode: The render mode.
    """
    metadata = {"render_modes": ["ansi", "human"]}
    size = GRID_SIZE
    observation_space = spaces.Box(low=-1,
                                   high=1,
                                   shape=(GRID_SIZE,),
                                   dtype=np.int8)
    action_space = spaces.Discrete(GRID_SIZE)

    def __init__(self, render_mode=None):
        initial_state: Connect2State = np.zeros(GRID_SIZE, dtype=np.int8)
        super().__init__(initial_state=initial_state)

        self.render_mode = render_mode

    @staticmethod
    def legal_moves(state: Connect2State) -> np.ndarray:
        """Returns the legal moves for the current player."""
        return np.where(state == 0)[0]

    def _get_info(self) -> dict[str, int]:
        """Returns the information for the current state."""
        return {"current_player": self.current_player}

    def check_action(self, state: Connect2State, action: int) -> bool:
        """Checks if the action is legal.

        Args:
            state: The current state.
            action: The action to take.

        Returns:
            True if the action is legal, False otherwise.

        Raises:
            ValueError: If the action is not in the action space.
        """
        if action not in self.action_space:
            raise ValueError(f"Invalid action: {action}")

        return state[action] == 0

    @staticmethod
    def is_finished(state: Connect2State) -> bool:
        """Checks if the game is finished."""
        return Connect2.is_winning_state(state) or np.all(state != 0)

    @staticmethod
    def get_current_player(state: Connect2State) -> int:
        """Returns the current player."""

        n_chips = np.sum(np.abs(state))
        return 1 if n_chips % 2 == 0 else -1

    @staticmethod
    def get_reward(state: Connect2State) -> int:
        """Returns the reward for the current state.

        0 if the game is not over, 1 if player 1 wins, and -1 if player 2 (
        encoded as player -1) wins.

        Args:
            state: The current state.
        """
        player = Connect2.get_current_player(state)
        if Connect2.is_winning_state(state):
            return -player
        return 0

    @staticmethod
    def is_winning_state(state: Connect2State) -> bool:
        """Check for two consecutive chips of the current player"""

        for i in range(Connect2.size - 1):
            if state[i] == state[i + 1] and state[i] != 0:
                return True
        return False

    @staticmethod
    def next_state(state: Connect2State, action: int) -> Connect2State:
        """Returns the next state given the current state and action."""
        next_state = state.copy()
        current_player = Connect2.get_current_player(state)
        next_state[action] = current_player
        return next_state

    def step(self, action: int) -> tuple[
        Connect2State, float, bool, bool, dict[str, Any]
    ]:
        """Takes a step in the environment.

        Args:
            action: The action to take. Must be an integer between 0 and 3.

        Returns:
            The next state, the reward, whether the episode is terminated,
            whether the episode is truncated, and the info.

        Raises:
            ValueError: If the action is invalid or illegal.
        """
        if action not in self.action_space:
            raise ValueError(f"Invalid action: {action}")

        if not self.check_action(self.current_state, action):
            raise ValueError(f"The action {action} is not legal. "
                             f"Current state: {self.current_state}")

        self.current_state = Connect2.next_state(self.current_state, action)
        terminated = Connect2.is_finished(self.current_state)
        reward = Connect2.get_reward(self.current_state)
        self.current_player = -self.current_player
        if self.render_mode == "human":
            self.render()

        return self.current_state, reward, terminated, False, self._get_info()

    def render(self) -> Connect2State:
        """Renders the environment."""
        if self.render_mode in ["ansi", "human"]:
            print(self.current_state)
            return self.current_state

        raise NotImplementedError(
            f"Render mode {self.render_mode} is not implemented.")
