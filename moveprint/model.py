"""DeepFM-inspired blunder prediction model (Architecture 14).

Faithful re-implementation of the winning architecture described in Section 3.1
and Table 1 of the paper. Component labels (a)-(g) below match the paper.

Inputs (a):
    board_planes  (B, NUM_INPUT_PLANES, 8, 8)  -> frozen CNN -> board embedding
    user_id       (B,)                          -> two dedicated embeddings (d)
    user_rating   (B,)  (optional, normalised)
    opp_rating    (B,)  (optional, normalised)
    ply           (B,)  (normalised)

Flow:
    (b) board embedding from the frozen RISEv2-mobile CNN.
    (c) board embedding -> FC(500) + ReLU.
    (d) user id -> two 25-d embeddings (deep + factorization).
    (e) DEEP: concat[board500, user_emb_deep, rating, opp_rating, ply]
            -> FC(64) -> FC(64) -> ReLU  => x
    (f) FACTORIZATION: board -> FC(25) + ReLU; inner product with user_emb_fm
            (element-wise product, summed) => y
    (g) HEAD: concat[x, y] -> FC(256) x4 -> FC(1) -> sigmoid.

`use_elo` toggles inclusion of the two rating features (ablation, Table 2).
`use_user_id` toggles user embeddings entirely (ablation: "no user information").
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .cnn import RISEv2Mobile, build_frozen_board_extractor
from .config import CNNConfig, ModelConfig


def _mlp(in_dim: int, units, activation=nn.ReLU, final_activation=True) -> nn.Sequential:
    layers = []
    prev = in_dim
    for i, u in enumerate(units):
        layers.append(nn.Linear(prev, u))
        if final_activation or i < len(units) - 1:
            layers.append(activation())
        prev = u
    return nn.Sequential(*layers)


class BlunderPredictor(nn.Module):
    def __init__(
        self,
        num_users: int,
        model_cfg: Optional[ModelConfig] = None,
        cnn_cfg: Optional[CNNConfig] = None,
        board_extractor: Optional[RISEv2Mobile] = None,
    ):
        super().__init__()
        self.cfg = model_cfg or ModelConfig()
        self.cnn_cfg = cnn_cfg or CNNConfig()
        self.num_users = num_users

        # (b) Frozen board extractor. Shared / reused; parameters are frozen.
        self.board_extractor = board_extractor or build_frozen_board_extractor(self.cnn_cfg)

        # (c) Board -> 500 features.
        self.board_fc = nn.Sequential(
            nn.Linear(self.cfg.board_embedding_dim, self.cfg.board_fc_units),
            nn.ReLU(inplace=True),
        )

        # (d) Two dedicated user embeddings (deep + factorization).
        if self.cfg.use_user_id:
            self.user_emb_deep = nn.Embedding(num_users, self.cfg.user_embedding_dim)
            self.user_emb_fm = nn.Embedding(num_users, self.cfg.user_embedding_dim)
            nn.init.normal_(self.user_emb_deep.weight, std=0.05)
            nn.init.normal_(self.user_emb_fm.weight, std=0.05)
        else:
            self.user_emb_deep = None
            self.user_emb_fm = None

        # Count of scalar metadata features feeding the deep component.
        # ply is always included; ratings only when use_elo.
        n_scalar = 1 + (2 if self.cfg.use_elo else 0)
        deep_in = self.cfg.board_fc_units + n_scalar
        if self.cfg.use_user_id:
            deep_in += self.cfg.user_embedding_dim

        # (e) Deep component.
        self.deep = _mlp(deep_in, self.cfg.deep_units, final_activation=True)
        deep_out = self.cfg.deep_units[-1]

        # (f) Factorization board projection.
        self.fm_board = nn.Sequential(
            nn.Linear(self.cfg.board_embedding_dim, self.cfg.fm_board_units),
            nn.ReLU(inplace=True),
        )
        # y is a scalar (inner product) when user id is used; else zero-width.
        fm_out = 1 if self.cfg.use_user_id else 0

        # (g) Joint head.
        head_in = deep_out + fm_out
        self.head_body = _mlp(head_in, self.cfg.head_units, final_activation=True)
        self.head_out = nn.Linear(self.cfg.head_units[-1], 1)

    # -- board embedding (frozen) -------------------------------------------
    @torch.no_grad()
    def _board_embedding(self, board_planes: torch.Tensor) -> torch.Tensor:
        self.board_extractor.eval()
        return self.board_extractor(board_planes)

    def forward(
        self,
        board_planes: torch.Tensor,
        user_id: torch.Tensor,
        user_rating: torch.Tensor,
        opp_rating: torch.Tensor,
        ply: torch.Tensor,
        precomputed_board_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return raw logits (B,). Apply sigmoid for probabilities."""
        if precomputed_board_emb is not None:
            board_emb = precomputed_board_emb
        else:
            board_emb = self._board_embedding(board_planes)

        board500 = self.board_fc(board_emb)  # (c)

        # Assemble deep-component input (e).
        deep_parts = [board500]
        if self.cfg.use_user_id:
            deep_parts.append(self.user_emb_deep(user_id))
        if self.cfg.use_elo:
            deep_parts.append(user_rating.unsqueeze(-1))
            deep_parts.append(opp_rating.unsqueeze(-1))
        deep_parts.append(ply.unsqueeze(-1))
        deep_in = torch.cat(deep_parts, dim=-1)
        x = self.deep(deep_in)

        # Factorization component (f): inner product of user_fm and board_fm.
        if self.cfg.use_user_id:
            fm_board = self.fm_board(board_emb)              # (B, 25)
            user_fm = self.user_emb_fm(user_id)              # (B, 25)
            y = (fm_board * user_fm).sum(dim=-1, keepdim=True)  # (B, 1)
            joint = torch.cat([x, y], dim=-1)
        else:
            joint = x

        # Head (g).
        h = self.head_body(joint)
        logits = self.head_out(h).squeeze(-1)
        return logits

    def trainable_parameters(self):
        """Parameters that require grad (excludes the frozen CNN)."""
        return [p for p in self.parameters() if p.requires_grad]

    def add_new_user(self) -> int:
        """Cold-start: grow the embedding tables for a new user.

        Returns the new user's id. Demonstrates the paper's claim that a new
        user only requires learning their embedding vector rather than a whole
        new model. The new embedding is initialised as the mean of existing
        embeddings (a reasonable warm start).
        """
        if not self.cfg.use_user_id:
            raise RuntimeError("Model does not use user embeddings.")
        new_id = self.num_users
        for emb_attr in ("user_emb_deep", "user_emb_fm"):
            old = getattr(self, emb_attr)
            new = nn.Embedding(self.num_users + 1, self.cfg.user_embedding_dim)
            with torch.no_grad():
                new.weight[: self.num_users] = old.weight
                new.weight[self.num_users] = old.weight.mean(dim=0)
            setattr(self, emb_attr, new.to(old.weight.device))
        self.num_users += 1
        return new_id
