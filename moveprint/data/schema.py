"""Dataset record definitions.

Each training/eval example corresponds to a single move a player is about to
make in a given board position, together with metadata and labels. This matches
the dataset described in Section 4 of the paper:

  * user id (player about to move)
  * board (arrangement of pieces)
  * user rating, opponent rating
  * ply (combined move count of both players)
  * blunder label, and if a blunder, its type (immediate / non-immediate)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np


class BlunderType(IntEnum):
    NONE = 0
    IMMEDIATE = 1        # tactical; consequence obvious immediately
    NON_IMMEDIATE = 2    # strategic; consequence appears in later moves


class GamePhase(IntEnum):
    OPENING = 0
    MIDDLE = 1
    ENDGAME = 2


@dataclass
class MoveRecord:
    """A single move example."""

    user_id: int                 # dense integer id of the player to move
    game_id: int                 # id of the game (for chronological ordering)
    move_index: int              # order within the player's game history
    board_planes: np.ndarray     # (NUM_INPUT_PLANES, 8, 8) float32
    user_rating: float
    opponent_rating: float
    ply: int                     # combined move count
    is_blunder: int              # 0/1
    blunder_type: BlunderType    # NONE / IMMEDIATE / NON_IMMEDIATE
    phase: GamePhase
    # Chronological key: earlier = smaller. Used for the 80/20 split.
    time_order: float = 0.0


@dataclass
class DatasetStats:
    """Summary counts (mirrors the numbers reported in Section 4)."""

    num_moves: int
    num_blunders: int
    num_immediate: int
    num_non_immediate: int

    def __str__(self) -> str:
        return (
            f"moves={self.num_moves} blunders={self.num_blunders} "
            f"immediate={self.num_immediate} non_immediate={self.num_non_immediate} "
            f"(blunder rate={self.num_blunders / max(1, self.num_moves):.3f})"
        )
