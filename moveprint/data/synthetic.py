"""Offline synthetic dataset generator.

Produces a class-imbalanced dataset (~8% blunders, matching the paper's stated
rate) with realistic-looking board tensors and per-player "blunder profiles" so
the *collaborative personalisation* claim can actually be exercised: players
with similar latent profiles blunder in correlated situations.

This lets the full pipeline (encoding -> frozen CNN -> DeepFM -> AUC/AUC-PR) run
end-to-end without the external Lichess corpus or Stockfish. It is NOT a
substitute for real evaluation; it exists to validate the implementation.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..config import NUM_INPUT_PLANES
from .schema import BlunderType, GamePhase, MoveRecord


def _phase_from_ply(ply: int) -> GamePhase:
    if ply < 20:
        return GamePhase.OPENING
    if ply < 60:
        return GamePhase.MIDDLE
    return GamePhase.ENDGAME


def generate_synthetic_dataset(
    num_players: int = 60,
    games_per_player: int = 20,
    moves_per_game: int = 30,
    latent_dim: int = 25,
    blunder_rate: float = 0.08,
    seed: int = 42,
) -> List[MoveRecord]:
    """Generate a synthetic set of MoveRecords.

    Each player has a latent "blunder profile". Each board carries a *motif*
    signal that is encoded directly into the board planes (what the frozen CNN
    actually sees). The blunder probability for a (player, board) pair is a
    logistic function of the interaction between the player's profile and the
    board's motif signal, plus a skill and a ply term.

    Because the signal lives in the board planes (not an invisible latent), it
    is recoverable through the frozen CNN -> DeepFM path, so the smoke test is a
    genuine learnability check. The logistic bias is calibrated empirically so
    the realised blunder rate is close to `blunder_rate` (~8%, per the paper).
    """
    rng = np.random.default_rng(seed)

    # Latent player profiles live in the *motif* space (dimension = n_motifs).
    n_motifs = 8
    player_profiles = rng.normal(0, 1, size=(num_players, n_motifs))
    # Player skill (rating) is correlated with a lower blunder propensity.
    player_skill = rng.normal(0, 1, size=num_players)
    player_rating = 1500 + 300 * player_skill + rng.normal(0, 80, size=num_players)
    player_rating = np.clip(player_rating, 600, 2500)

    def raw_logit(profile, skill, motif_vec, ply):
        interaction = float(profile @ motif_vec) / np.sqrt(n_motifs)
        ply_effect = 0.15 * np.sin(ply / 6.0)
        return 1.6 * interaction - 0.9 * skill + ply_effect

    # --- Calibrate the bias so mean(sigmoid(bias + raw_logit)) ~= blunder_rate.
    sample_logits = []
    cal_rng = np.random.default_rng(seed + 1)
    for _ in range(4000):
        uid = cal_rng.integers(0, num_players)
        motif_vec, _ = _sample_motifs(cal_rng, n_motifs)
        ply = cal_rng.integers(0, 2 * moves_per_game)
        sample_logits.append(raw_logit(player_profiles[uid], player_skill[uid], motif_vec, ply))
    sample_logits = np.array(sample_logits)

    def rate_at(bias):
        return float(np.mean(1.0 / (1.0 + np.exp(-(bias + sample_logits)))))

    lo, hi = -12.0, 12.0
    for _ in range(60):  # bisection on the bias
        mid = 0.5 * (lo + hi)
        if rate_at(mid) < blunder_rate:
            lo = mid
        else:
            hi = mid
    bias = 0.5 * (lo + hi)

    records: List[MoveRecord] = []
    global_game_id = 0

    for uid in range(num_players):
        profile = player_profiles[uid]
        skill = player_skill[uid]
        for g in range(games_per_player):
            global_game_id += 1
            opp_rating = float(np.clip(player_rating[uid] + rng.normal(0, 150), 600, 2500))
            time_order = float(g)  # chronological within player
            for m in range(moves_per_game):
                ply = 2 * m + rng.integers(0, 2)
                phase = _phase_from_ply(ply)

                # Motif vector + the planes that encode it (what the CNN sees).
                motif_vec, planes = _sample_motifs(rng, n_motifs)

                logit = bias + raw_logit(profile, skill, motif_vec, ply)
                p_blunder = 1.0 / (1.0 + np.exp(-logit))
                is_blunder = int(rng.random() < p_blunder)

                if is_blunder:
                    # Less-skilled players lean immediate (tactical); skilled
                    # players lean non-immediate (strategic). Mirrors Fig. 4.
                    p_immediate = 1.0 / (1.0 + np.exp(1.2 * skill))
                    btype = (
                        BlunderType.IMMEDIATE
                        if rng.random() < p_immediate
                        else BlunderType.NON_IMMEDIATE
                    )
                else:
                    btype = BlunderType.NONE

                records.append(
                    MoveRecord(
                        user_id=uid,
                        game_id=global_game_id,
                        move_index=m,
                        board_planes=planes,
                        user_rating=float(player_rating[uid]),
                        opponent_rating=opp_rating,
                        ply=int(ply),
                        is_blunder=is_blunder,
                        blunder_type=btype,
                        phase=phase,
                        time_order=time_order,
                    )
                )

    rng.shuffle(records)
    return records


def _sample_motifs(rng: np.random.Generator, n_motifs: int):
    """Sample a motif activation vector and encode it into board planes.

    Each motif corresponds to a fixed 2x2 block location on the first few piece
    planes. An active motif stamps its block, so the motif signal is literally
    present in the planes the frozen CNN consumes. Returns (motif_vec, planes).
    """
    planes = np.zeros((NUM_INPUT_PLANES, 8, 8), dtype=np.float32)
    motif_vec = np.zeros(n_motifs, dtype=np.float32)

    for k in range(n_motifs):
        active = rng.random() < 0.5
        motif_vec[k] = 1.0 if active else 0.0
        if active:
            plane = k % 12
            r = (k * 2) % 7
            c = (k * 3) % 7
            planes[plane, r:r + 2, c:c + 2] = 1.0

    # Add light background clutter so inputs are not purely the motif blocks.
    for plane in range(12):
        n = rng.integers(0, 3)
        for _ in range(n):
            r = rng.integers(0, 8)
            c = rng.integers(0, 8)
            planes[plane, r, c] = 1.0
    # Side to move + a couple of auxiliary planes.
    planes[12, :, :] = float(rng.integers(0, 2))
    for plane in range(13, 17):
        planes[plane, :, :] = float(rng.integers(0, 2))
    planes[18, :, :] = rng.random() * 0.5
    planes[19, :, :] = rng.random() * 0.6
    return motif_vec, planes
