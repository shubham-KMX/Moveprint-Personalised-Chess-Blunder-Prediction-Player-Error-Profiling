"""Real-data ingestion from PGN files (Lichess-style).

Builds MoveRecords from PGN games. Each ply where a target player is to move
becomes a candidate example. Blunder labelling requires engine evaluations:

  * If a UCI engine (e.g. Stockfish) is available on PATH, positions before and
    after each move are evaluated and labelled with the win-probability-drop
    rule from `labeling.py`. Immediate/non-immediate classification uses a short
    engine look-ahead.
  * If no engine is available, PGN NAG annotations ($4 = blunder) or `??`
    comments are used as a coarse fallback label, and blunder-type defaults to
    NON_IMMEDIATE unless a mate is annotated.

This module is intentionally dependency-guarded so importing the package does
not require python-chess to be installed.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from ..config import NUM_INPUT_PLANES
from .labeling import label_move
from .schema import BlunderType, GamePhase, MoveRecord

try:
    import chess
    import chess.engine
    import chess.pgn

    _HAS_CHESS = True
except Exception:  # pragma: no cover
    _HAS_CHESS = False


# NAGs indicating a blunder in PGN annotations.
_BLUNDER_NAGS = {4, 9}  # $4 = very poor move (??), $9 = very poor move


def _phase_of(board: "chess.Board") -> GamePhase:
    """Game phase per the Lichess definition used in the paper (Section 5.6)."""
    # Endgame: total non-king, non-pawn pieces <= 6.
    minor_major = 0
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        minor_major += len(board.pieces(pt, chess.WHITE))
        minor_major += len(board.pieces(pt, chess.BLACK))
    if minor_major <= 6:
        return GamePhase.ENDGAME
    if board.fullmove_number <= 10:
        return GamePhase.OPENING
    return GamePhase.MIDDLE


def _material(board: "chess.Board", color: bool) -> float:
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
              chess.ROOK: 5, chess.QUEEN: 9}
    total = 0.0
    for pt, v in values.items():
        total += v * len(board.pieces(pt, color))
    return total


def ingest_pgn_dir(
    pgn_dir: str,
    engine_path: Optional[str] = None,
    max_games: Optional[int] = None,
    engine_depth: int = 12,
) -> List[MoveRecord]:
    """Ingest all .pgn files in a directory into MoveRecords.

    Player ids are assigned densely by first appearance.
    """
    if not _HAS_CHESS:
        raise RuntimeError("python-chess is required for PGN ingestion.")

    from ..board import encode_board  # local import to avoid hard dependency

    engine = None
    if engine_path and os.path.exists(engine_path):
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    user_ids: Dict[str, int] = {}
    records: List[MoveRecord] = []
    game_counter = 0

    try:
        for fname in sorted(os.listdir(pgn_dir)):
            if not fname.lower().endswith(".pgn"):
                continue
            with open(os.path.join(pgn_dir, fname), "r", encoding="utf-8", errors="ignore") as fh:
                while True:
                    game = chess.pgn.read_game(fh)
                    if game is None:
                        break
                    game_counter += 1
                    if max_games is not None and game_counter > max_games:
                        break
                    records.extend(
                        _records_from_game(game, game_counter, user_ids, engine, engine_depth)
                    )
    finally:
        if engine is not None:
            engine.quit()

    return records


def _records_from_game(game, game_id, user_ids, engine, depth) -> List[MoveRecord]:
    from ..board import encode_board

    white = game.headers.get("White", "unknown_white")
    black = game.headers.get("Black", "unknown_black")
    try:
        white_elo = float(game.headers.get("WhiteElo", "1500"))
        black_elo = float(game.headers.get("BlackElo", "1500"))
    except ValueError:
        white_elo, black_elo = 1500.0, 1500.0

    for name in (white, black):
        if name not in user_ids:
            user_ids[name] = len(user_ids)

    # Use UTCDate/UTCTime for chronology if present.
    date = game.headers.get("UTCDate", "0000.00.00")
    time_order = float(game_id)  # fallback ordering
    try:
        time_order = float(date.replace(".", ""))
    except ValueError:
        pass

    board = game.board()
    records: List[MoveRecord] = []
    move_index = 0
    node = game

    while node.variations:
        next_node = node.variations[0]
        move = next_node.move
        mover_is_white = board.turn == chess.WHITE
        mover_name = white if mover_is_white else black
        mover_elo = white_elo if mover_is_white else black_elo
        opp_elo = black_elo if mover_is_white else white_elo

        planes = encode_board(board)
        phase = _phase_of(board)
        ply = board.ply()

        is_blunder, btype = _label(board, next_node, mover_is_white, engine, depth)

        records.append(
            MoveRecord(
                user_id=user_ids[mover_name],
                game_id=game_id,
                move_index=move_index,
                board_planes=planes,
                user_rating=mover_elo,
                opponent_rating=opp_elo,
                ply=ply,
                is_blunder=is_blunder,
                blunder_type=btype,
                phase=phase,
                time_order=time_order,
            )
        )
        board.push(move)
        node = next_node
        move_index += 1

    return records


def _label(board, next_node, mover_is_white, engine, depth):
    """Label a single move as (is_blunder, BlunderType)."""
    if engine is not None:
        return _label_with_engine(board, next_node.move, mover_is_white, engine, depth)
    # Fallback: PGN NAG annotations.
    nags = getattr(next_node, "nags", set())
    if _BLUNDER_NAGS & set(nags):
        return 1, BlunderType.NON_IMMEDIATE
    return 0, BlunderType.NONE


def _label_with_engine(board, move, mover_is_white, engine, depth):
    import chess.engine

    def score_cp(b):
        info = engine.analyse(b, chess.engine.Limit(depth=depth))
        return info["score"].white().score(mate_score=2000)

    cp_before = score_cp(board)
    after = board.copy()
    after.push(move)
    cp_after = score_cp(after)

    material_before = _material(board, chess.WHITE if mover_is_white else chess.BLACK)

    # Best opponent reply.
    ends_mate = False
    material_best_reply = 0.0
    material_second = 0.0
    if not after.is_game_over():
        reply_info = engine.play(after, chess.engine.Limit(depth=depth))
        reply = reply_info.move
        after2 = after.copy()
        after2.push(reply)
        ends_mate = after2.is_checkmate()
        mat = _material(after2, chess.WHITE if mover_is_white else chess.BLACK)
        material_best_reply = mat - material_before
        # One more ply for the non-immediate check.
        if not after2.is_game_over():
            follow = engine.play(after2, chess.engine.Limit(depth=depth)).move
            after3 = after2.copy()
            after3.push(follow)
            if not after3.is_game_over():
                second = engine.play(after3, chess.engine.Limit(depth=depth)).move
                after4 = after3.copy()
                after4.push(second)
                material_second = _material(after4, chess.WHITE if mover_is_white else chess.BLACK) - material_before
    else:
        ends_mate = after.is_checkmate()

    return label_move(
        cp_before,
        cp_after,
        mover_is_white,
        material_delta_after_best_reply=material_best_reply,
        material_delta_after_second_move=material_second,
        ends_in_checkmate_immediately=ends_mate,
    )
