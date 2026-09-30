"""Board -> input-plane encoding for the frozen CNN board feature extractor.

We encode a chess position into a stack of 8x8 planes, following the standard
AlphaZero / CrazyAra convention (piece planes + auxiliary state planes). The
resulting tensor has shape (NUM_INPUT_PLANES, 8, 8) and preserves the spatial
board layout, which the RISEv2-mobile CNN consumes without any pooling.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:  # python-chess is optional at import time so the module can be introspected
    import chess

    _HAS_CHESS = True
except Exception:  # pragma: no cover - environment without python-chess
    chess = None  # type: ignore
    _HAS_CHESS = False

from .config import BOARD_SIZE, NUM_INPUT_PLANES

# Ordering of piece types for the 12 piece planes (white first, then black).
_PIECE_ORDER = None
if _HAS_CHESS:
    _PIECE_ORDER = [
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    ]


def encode_board(board: "chess.Board") -> np.ndarray:
    """Encode a python-chess Board into a (NUM_INPUT_PLANES, 8, 8) float32 array.

    Plane layout (20 planes):
      0-5   white P,N,B,R,Q,K
      6-11  black P,N,B,R,Q,K
      12    side to move (1 if white to move)
      13-16 castling rights: WK, WQ, BK, BQ
      17    en-passant target square
      18    halfmove clock (scaled by /100)
      19    fullmove number (scaled by /100)

    Planes are always laid out from White's perspective (rank 1 at row 0).
    """
    if not _HAS_CHESS:
        raise RuntimeError("python-chess is required to encode a real board.")

    planes = np.zeros((NUM_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    for square, piece in board.piece_map().items():
        row = chess.square_rank(square)
        col = chess.square_file(square)
        piece_idx = _PIECE_ORDER.index(piece.piece_type)
        plane = piece_idx if piece.color == chess.WHITE else piece_idx + 6
        planes[plane, row, col] = 1.0

    if board.turn == chess.WHITE:
        planes[12, :, :] = 1.0

    planes[13, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
    planes[14, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
    planes[15, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
    planes[16, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))

    if board.ep_square is not None:
        row = chess.square_rank(board.ep_square)
        col = chess.square_file(board.ep_square)
        planes[17, row, col] = 1.0

    planes[18, :, :] = min(board.halfmove_clock, 100) / 100.0
    planes[19, :, :] = min(board.fullmove_number, 100) / 100.0

    return planes


def encode_fen(fen: str) -> np.ndarray:
    """Encode a FEN string into input planes."""
    if not _HAS_CHESS:
        raise RuntimeError("python-chess is required to encode a FEN.")
    return encode_board(chess.Board(fen))


def empty_planes() -> np.ndarray:
    """Return an all-zero plane stack (useful for padding / synthetic paths)."""
    return np.zeros((NUM_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
