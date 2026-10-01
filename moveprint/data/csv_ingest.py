"""Ingestion for the pre-labelled moves CSV (real data).

Expected schema (as produced by the user's PGN -> FEN -> Stockfish pipeline):

    game_id, player_id, player_elo, opponent_elo, move_ply, board_fen,
    move_uci, is_blunder, blunder_type, wp_drop

Each row is one move the given player is about to make, already labelled with a
Stockfish win-probability-drop blunder flag (>= 10% drop). This maps directly to
`MoveRecord` (Section 4 of the paper).

Chronological order (for the per-player 80/20 split) is taken from the row order
in the file: games appear in play order and plies are monotonic within a game,
so a running counter is a faithful chronological key.

`blunder_type` may be one of:
    none / blunder                      -> only generic blunder labels available
    none / immediate / non_immediate    -> full per-type labels available
"""

from __future__ import annotations

import csv
from typing import Dict, List, Optional

import numpy as np

from .schema import BlunderType, GamePhase, MoveRecord

# Allow large FEN/boards fields.
csv.field_size_limit(10_000_000)

_BLUNDER_TYPE_MAP = {
    "none": BlunderType.NONE,
    "": BlunderType.NONE,
    "immediate": BlunderType.IMMEDIATE,
    "non_immediate": BlunderType.NON_IMMEDIATE,
    "non-immediate": BlunderType.NON_IMMEDIATE,
    "nonimmediate": BlunderType.NON_IMMEDIATE,
    # Datasets that only distinguish blunder / non-blunder label a generic
    # blunder as "blunder"; we map it to NON_IMMEDIATE as a neutral default so
    # it is still counted as a blunder (type-specific models require the finer
    # labels and should be skipped when only this coarse label exists).
    "blunder": BlunderType.NON_IMMEDIATE,
}


def _phase_from_fen(fen: str, fullmove: int) -> GamePhase:
    """Game phase using the piece-count rule from the paper (Section 5.6).

    Endgame: total non-king, non-pawn pieces <= 6. Opening: fullmove <= 10.
    Otherwise middle game. Computed from the board part of the FEN so python-chess
    is not required for ingestion.
    """
    board_part = fen.split(" ", 1)[0]
    minor_major = sum(1 for ch in board_part if ch in "qrbnQRBN")
    if minor_major <= 6:
        return GamePhase.ENDGAME
    if fullmove <= 10:
        return GamePhase.OPENING
    return GamePhase.MIDDLE


def _fullmove_from_fen(fen: str) -> int:
    parts = fen.split(" ")
    try:
        return int(parts[5])
    except (IndexError, ValueError):
        return 1


def has_fine_grained_types(csv_path: str, sample: int = 5000) -> bool:
    """Return True if the CSV contains immediate / non_immediate labels."""
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            bt = (row.get("blunder_type") or "").strip().lower()
            if bt in ("immediate", "non_immediate", "non-immediate", "nonimmediate"):
                return True
            if i >= sample:
                break
    return False


def ingest_moves_csv(
    csv_path: str,
    max_rows: Optional[int] = None,
    encode_boards: bool = True,
) -> List[MoveRecord]:
    """Read the labelled moves CSV into MoveRecords.

    Parameters
    ----------
    csv_path:
        Path to moves_labeled.csv.
    max_rows:
        Optional cap on the number of rows read (useful for quick runs).
    encode_boards:
        If True, encode each FEN into input planes eagerly. For very large files
        this uses significant memory; set False to defer (planes left as a small
        placeholder) — but the model needs planes, so keep True for training.
    """
    from ..board import encode_fen  # local import; requires python-chess

    player_ids: Dict[str, int] = {}
    game_ids: Dict[str, int] = {}
    records: List[MoveRecord] = []

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break

            player_name = row["player_id"]
            if player_name not in player_ids:
                player_ids[player_name] = len(player_ids)
            uid = player_ids[player_name]

            game_name = row["game_id"]
            if game_name not in game_ids:
                game_ids[game_name] = len(game_ids)
            gid = game_ids[game_name]

            fen = row["board_fen"]
            fullmove = _fullmove_from_fen(fen)
            phase = _phase_from_fen(fen, fullmove)

            if encode_boards:
                planes = encode_fen(fen)
            else:
                planes = np.zeros((1, 1, 1), dtype=np.float32)

            is_blunder = int(float(row.get("is_blunder", 0)))
            bt_raw = (row.get("blunder_type") or "").strip().lower()
            btype = _BLUNDER_TYPE_MAP.get(bt_raw, BlunderType.NONE)
            if not is_blunder:
                btype = BlunderType.NONE

            try:
                user_rating = float(row.get("player_elo", 1500) or 1500)
            except ValueError:
                user_rating = 1500.0
            try:
                opp_rating = float(row.get("opponent_elo", 1500) or 1500)
            except ValueError:
                opp_rating = 1500.0
            try:
                ply = int(float(row.get("move_ply", 0) or 0))
            except ValueError:
                ply = 0

            records.append(
                MoveRecord(
                    user_id=uid,
                    game_id=gid,
                    move_index=ply,
                    board_planes=planes,
                    user_rating=user_rating,
                    opponent_rating=opp_rating,
                    ply=ply,
                    is_blunder=is_blunder,
                    blunder_type=btype,
                    phase=phase,
                    time_order=float(i),  # file order == chronological proxy
                )
            )

    return records
