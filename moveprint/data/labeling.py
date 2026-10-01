"""Blunder labelling and immediate / non-immediate classification.

Definitions from the paper (Sections 1 and 4), following McIlroy-Young 2020:

  * Each board position is annotated with an engine score (e.g. Stockfish),
    mapped to a *win probability*. A move is a **blunder** if the win
    probability drops by at least 10% after the move.

  * **Immediate blunder**: after the blunder move, the opponent responds with
    the best move and the game ends in checkmate, OR the blundering player also
    plays the best reply and still suffers a material loss relative to the
    position before the blunder. (Consequence is immediate.)

  * **Non-immediate blunder**: the material loss / loss of the game only
    materialises on the opponent's *second* move after the blunder or later.
    (Consequence is delayed / strategic.)

This module provides both:
  * `win_prob` / `is_blunder_from_scores` — pure win-probability logic that works
    with engine centipawn scores.
  * `classify_blunder_type` — the immediate/non-immediate decision given a short
    look-ahead of engine scores and material deltas.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

from ..config import BLUNDER_WINPROB_DROP, MATERIAL_LOSS_THRESHOLD
from .schema import BlunderType


def win_prob(centipawns: float, k: float = 0.00368208) -> float:
    """Map an engine centipawn score to a win probability in [0, 1].

    Uses the standard logistic mapping used by Lichess / prior chess ML work:
        wp = 1 / (1 + exp(-k * cp))
    The constant k corresponds to the widely used 1/(1+10^(-cp/400)) form.
    """
    # Clamp to avoid overflow on mate scores expressed as large centipawns.
    cp = max(-2000.0, min(2000.0, centipawns))
    return 1.0 / (1.0 + math.exp(-k * cp))


def is_blunder_from_scores(cp_before: float, cp_after: float,
                           mover_is_white: bool,
                           threshold: float = BLUNDER_WINPROB_DROP) -> bool:
    """Return True if the move dropped the mover's win probability by >= threshold.

    Engine scores are from White's perspective (positive = good for White).
    We convert to the *mover's* win probability before comparing.
    """
    wp_before = win_prob(cp_before)
    wp_after = win_prob(cp_after)
    if not mover_is_white:
        wp_before = 1.0 - wp_before
        wp_after = 1.0 - wp_after
    return (wp_before - wp_after) >= threshold


def classify_blunder_type(
    cp_before: float,
    material_delta_after_best_reply: float,
    material_delta_after_second_move: float,
    ends_in_checkmate_immediately: bool,
    mover_is_white: bool,
    material_threshold: float = MATERIAL_LOSS_THRESHOLD,
) -> BlunderType:
    """Classify a blunder as IMMEDIATE or NON_IMMEDIATE.

    Parameters
    ----------
    cp_before:
        Engine score before the blunder (White's perspective). Unused directly
        but kept for interface completeness / future extensions.
    material_delta_after_best_reply:
        Material change (in pawns, from the mover's perspective, negative = loss)
        measured after the opponent plays the best reply and the mover plays the
        best follow-up. Used for the immediate condition.
    material_delta_after_second_move:
        Material change measured on the opponent's *second* move after the
        blunder or onwards. Used for the non-immediate condition.
    ends_in_checkmate_immediately:
        True if the best opponent reply delivers checkmate.
    """
    # Immediate: checkmate on the best reply, or material already lost after the
    # best reply + best follow-up.
    if ends_in_checkmate_immediately:
        return BlunderType.IMMEDIATE
    if material_delta_after_best_reply <= -material_threshold:
        return BlunderType.IMMEDIATE
    # Otherwise the damage is realised later -> strategic / non-immediate.
    if material_delta_after_second_move <= -material_threshold:
        return BlunderType.NON_IMMEDIATE
    # Falls back to non-immediate for a win-prob blunder without an obvious
    # immediate material swing.
    return BlunderType.NON_IMMEDIATE


def label_move(
    cp_before: float,
    cp_after: float,
    mover_is_white: bool,
    material_delta_after_best_reply: float = 0.0,
    material_delta_after_second_move: float = 0.0,
    ends_in_checkmate_immediately: bool = False,
) -> tuple[int, BlunderType]:
    """Full labelling for a single move.

    Returns (is_blunder, blunder_type).
    """
    if not is_blunder_from_scores(cp_before, cp_after, mover_is_white):
        return 0, BlunderType.NONE
    btype = classify_blunder_type(
        cp_before,
        material_delta_after_best_reply,
        material_delta_after_second_move,
        ends_in_checkmate_immediately,
        mover_is_white,
    )
    return 1, btype
