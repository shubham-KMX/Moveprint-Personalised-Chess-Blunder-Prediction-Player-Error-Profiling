"""Chronological train/test split and subset100 / subset750 construction.

From Section 4 of the paper:

  * Chronological split per player: the most recent 20% of each player's games
    are held out for evaluation; the earlier 80% are used for training. This
    prevents leakage and tests generalisation to future games.

  * subset100: the 100 players who made the most blunders (used for architecture
    search / tuning).

  * subset750: 750 players chosen by a full-factorial design over three factors
    (rating, blunder count, blunder-to-non-blunder ratio). Each factor is split
    into 5 equal-frequency bins -> 125 combinations; 6 players are sampled per
    combination -> 750 players.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .schema import MoveRecord


def chronological_split(
    records: Sequence[MoveRecord], test_fraction: float = 0.20
) -> Tuple[List[MoveRecord], List[MoveRecord]]:
    """Split per player: most recent `test_fraction` of games -> test.

    Games are ordered by (time_order, game_id). We hold out whole games (not
    individual moves) so that a game's moves never straddle the split for the
    same player.
    """
    # Group games per player, tracking each game's earliest time_order.
    player_games: Dict[int, Dict[int, float]] = defaultdict(dict)
    for r in records:
        cur = player_games[r.user_id].get(r.game_id)
        if cur is None or r.time_order < cur:
            player_games[r.user_id][r.game_id] = r.time_order

    # Determine, per player, which games are test games.
    test_games: Dict[int, set] = {}
    for uid, games in player_games.items():
        ordered = sorted(games.items(), key=lambda kv: (kv[1], kv[0]))
        n_test = max(1, int(round(len(ordered) * test_fraction))) if len(ordered) > 1 else 0
        test_ids = {gid for gid, _ in ordered[len(ordered) - n_test:]} if n_test else set()
        test_games[uid] = test_ids

    train: List[MoveRecord] = []
    test: List[MoveRecord] = []
    for r in records:
        if r.game_id in test_games.get(r.user_id, set()):
            test.append(r)
        else:
            train.append(r)
    return train, test


def _equal_frequency_bins(values: np.ndarray, num_bins: int) -> np.ndarray:
    """Assign each value to an equal-frequency bin index in [0, num_bins)."""
    if len(values) == 0:
        return np.array([], dtype=int)
    ranks = values.argsort().argsort()  # rank of each element
    # Map ranks -> bins so each bin holds ~equal count.
    bins = (ranks * num_bins / len(values)).astype(int)
    return np.clip(bins, 0, num_bins - 1)


def _player_factor_table(records: Sequence[MoveRecord]) -> Dict[int, Dict[str, float]]:
    """Compute per-player rating, blunder count, and blunder ratio."""
    stats: Dict[int, Dict[str, float]] = {}
    per_player: Dict[int, List[MoveRecord]] = defaultdict(list)
    for r in records:
        per_player[r.user_id].append(r)

    for uid, recs in per_player.items():
        n_blunder = sum(r.is_blunder for r in recs)
        n_non = len(recs) - n_blunder
        ratio = n_blunder / max(1, n_non)
        rating = float(np.mean([r.user_rating for r in recs]))
        stats[uid] = {
            "rating": rating,
            "blunders": float(n_blunder),
            "ratio": ratio,
            "moves": float(len(recs)),
        }
    return stats


def select_subset100(records: Sequence[MoveRecord], size: int = 100) -> List[int]:
    """Return the `size` player ids with the most blunders."""
    stats = _player_factor_table(records)
    ordered = sorted(stats.items(), key=lambda kv: kv[1]["blunders"], reverse=True)
    return [uid for uid, _ in ordered[:size]]


def select_subset750(
    records: Sequence[MoveRecord],
    size: int = 750,
    num_bins: int = 5,
    per_cell: int = 6,
    seed: int = 42,
) -> List[int]:
    """Full-factorial sampling over (rating, blunder count, ratio).

    Each factor -> `num_bins` equal-frequency bins => num_bins^3 cells. Sample
    `per_cell` players per cell. When a cell has fewer players than `per_cell`,
    it contributes what it has (the synthetic data or real data may be uneven).
    """
    rng = np.random.default_rng(seed)
    stats = _player_factor_table(records)
    if not stats:
        return []

    uids = np.array(list(stats.keys()))
    rating = np.array([stats[u]["rating"] for u in uids])
    blunders = np.array([stats[u]["blunders"] for u in uids])
    ratio = np.array([stats[u]["ratio"] for u in uids])

    b_rating = _equal_frequency_bins(rating, num_bins)
    b_blunder = _equal_frequency_bins(blunders, num_bins)
    b_ratio = _equal_frequency_bins(ratio, num_bins)

    cells: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for i, u in enumerate(uids):
        cells[(b_rating[i], b_blunder[i], b_ratio[i])].append(int(u))

    selected: List[int] = []
    for cell, members in cells.items():
        members = list(members)
        rng.shuffle(members)
        selected.extend(members[:per_cell])

    # If the full-factorial design yields more/less than requested, trim/pad.
    if len(selected) > size:
        rng.shuffle(selected)
        selected = selected[:size]
    return selected


def filter_by_players(records: Sequence[MoveRecord], player_ids: Sequence[int]) -> List[MoveRecord]:
    keep = set(player_ids)
    return [r for r in records if r.user_id in keep]
