"""Torch Dataset wrappers and feature normalisation.

Turns a list of MoveRecords into tensors, applying standard scaling to the
rating and ply inputs (component (a) in the paper: "players rating ... and move
ply inputs undergo normalization using standard scaling").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .data.schema import BlunderType, MoveRecord


@dataclass
class Normalizer:
    """Standard scaler fit on the training set."""

    rating_mean: float = 1500.0
    rating_std: float = 300.0
    ply_mean: float = 40.0
    ply_std: float = 20.0

    @classmethod
    def fit(cls, records: Sequence[MoveRecord]) -> "Normalizer":
        ratings = np.array(
            [r.user_rating for r in records] + [r.opponent_rating for r in records],
            dtype=np.float64,
        )
        plies = np.array([r.ply for r in records], dtype=np.float64)
        return cls(
            rating_mean=float(ratings.mean()),
            rating_std=float(ratings.std() + 1e-6),
            ply_mean=float(plies.mean()),
            ply_std=float(plies.std() + 1e-6),
        )

    def rating(self, x: float) -> float:
        return (x - self.rating_mean) / self.rating_std

    def ply(self, x: float) -> float:
        return (x - self.ply_mean) / self.ply_std


def _target_for(record: MoveRecord, target_type: Optional[BlunderType]) -> float:
    """Binary target.

    * target_type is None  -> generic blunder prediction (any blunder = 1).
    * target_type set       -> per-type model: 1 only for that blunder type,
      0 otherwise (even if it's a blunder of a different type). Matches
      Section 3.2.1 / 5.4.2.
    """
    if target_type is None:
        return float(record.is_blunder)
    return float(record.is_blunder and record.blunder_type == target_type)


class MoveDataset(Dataset):
    def __init__(
        self,
        records: Sequence[MoveRecord],
        normalizer: Normalizer,
        target_type: Optional[BlunderType] = None,
        board_embeddings: Optional[np.ndarray] = None,
    ):
        """`board_embeddings`, when given, is an (N, D) array aligned to
        `records` holding precomputed *frozen-CNN* board embeddings. When
        present, the raw planes are not returned and the model's CNN is skipped
        at train/eval time (the CNN is frozen, so this is exact and much
        faster).
        """
        self.records = list(records)
        self.norm = normalizer
        self.target_type = target_type
        self.board_embeddings = board_embeddings
        if board_embeddings is not None and len(board_embeddings) != len(self.records):
            raise ValueError("board_embeddings length must match records length")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        item = {
            "user_id": torch.tensor(r.user_id, dtype=torch.long),
            "user_rating": torch.tensor(self.norm.rating(r.user_rating), dtype=torch.float32),
            "opp_rating": torch.tensor(self.norm.rating(r.opponent_rating), dtype=torch.float32),
            "ply": torch.tensor(self.norm.ply(r.ply), dtype=torch.float32),
            "target": torch.tensor(_target_for(r, self.target_type), dtype=torch.float32),
            "phase": torch.tensor(int(r.phase), dtype=torch.long),
        }
        if self.board_embeddings is not None:
            item["board_emb"] = torch.from_numpy(
                np.ascontiguousarray(self.board_embeddings[idx])
            ).float()
        else:
            item["board_planes"] = torch.from_numpy(
                np.ascontiguousarray(r.board_planes)
            ).float()
        return item


def collate(batch):
    out = {
        "user_id": torch.stack([b["user_id"] for b in batch]),
        "user_rating": torch.stack([b["user_rating"] for b in batch]),
        "opp_rating": torch.stack([b["opp_rating"] for b in batch]),
        "ply": torch.stack([b["ply"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "phase": torch.stack([b["phase"] for b in batch]),
    }
    if "board_emb" in batch[0]:
        out["board_emb"] = torch.stack([b["board_emb"] for b in batch])
    else:
        out["board_planes"] = torch.stack([b["board_planes"] for b in batch])
    return out
