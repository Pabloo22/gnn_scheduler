"""Contains an implementation of Monte Carlo Tree Search algorithm."""
from __future__ import annotations

import numpy as np
import attrs

from alphazero.base import Game, State


@attrs.define
class Node:
    """Node in the Monte Carlo Tree Search tree."""

    state: State
    parent: 'Node' = attrs.field(default=None)
    player_to_move: int = attrs.field(default=1)
    children: list['Node'] = attrs.Factory(list)
    visits: int = 0
    value: float = 0.0


class MCTS:
    """Monte Carlo Tree Search"""

    def __init__(self, env: Game):
        self.env = env
        self.root = None

