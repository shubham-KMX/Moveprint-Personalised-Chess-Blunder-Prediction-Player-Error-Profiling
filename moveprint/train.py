"""Training loop with early stopping on validation AUC.

Implements Section 3/4:
  * Adam optimiser, lr 1e-4.
  * Weighted BCE (non-blunder weight 0.08).
  * Early stopping when validation AUC does not improve for 3 consecutive epochs.
  * Only trainable parameters are optimised (the CNN is frozen).
"""

from __future__ import annotations

import copy
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import TrainConfig
from .data.schema import BlunderType, MoveRecord
from .dataset import MoveDataset, Normalizer, collate
from .evaluate import evaluate
from .losses import weighted_bce
from .model import BlunderPredictor


def _split_train_val_idx(n: int, val_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(n * val_fraction)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


def train_model(
    model: BlunderPredictor,
    train_records: Sequence[MoveRecord],
    normalizer: Normalizer,
    cfg: Optional[TrainConfig] = None,
    device: Optional[torch.device] = None,
    target_type: Optional[BlunderType] = None,
    verbose: bool = True,
    board_embeddings: Optional[np.ndarray] = None,
) -> BlunderPredictor:
    """Train the model.

    If `board_embeddings` (an (N, D) array aligned to `train_records`) is
    provided, training uses the cached frozen-CNN embeddings and skips the CNN
    forward pass. This is exact (the CNN is frozen) and much faster.
    """
    cfg = cfg or TrainConfig()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    model = model.to(device)

    train_records = list(train_records)
    tr_idx, val_idx = _split_train_val_idx(len(train_records), cfg.val_fraction, cfg.seed)
    tr_records = [train_records[i] for i in tr_idx]
    val_records = [train_records[i] for i in val_idx]

    tr_emb = board_embeddings[tr_idx] if board_embeddings is not None else None
    val_emb = board_embeddings[val_idx] if board_embeddings is not None else None

    train_ds = MoveDataset(tr_records, normalizer, target_type=target_type, board_embeddings=tr_emb)
    val_ds = MoveDataset(val_records, normalizer, target_type=target_type, board_embeddings=val_emb)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)

    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=cfg.learning_rate)

    best_auc = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        # Keep the frozen extractor in eval mode (BatchNorm stats fixed).
        model.board_extractor.eval()

        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            optimizer.zero_grad()
            if "board_emb" in batch:
                logits = model(
                    None,
                    batch["user_id"].to(device),
                    batch["user_rating"].to(device),
                    batch["opp_rating"].to(device),
                    batch["ply"].to(device),
                    precomputed_board_emb=batch["board_emb"].to(device),
                )
            else:
                logits = model(
                    batch["board_planes"].to(device),
                    batch["user_id"].to(device),
                    batch["user_rating"].to(device),
                    batch["opp_rating"].to(device),
                    batch["ply"].to(device),
                )
            loss = weighted_bce(logits, batch["target"].to(device), cfg.non_blunder_weight)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1

        val_scores = evaluate(model, val_loader, device)
        val_auc = val_scores["auc"]
        if verbose:
            print(
                f"epoch {epoch:02d} | train_loss {total_loss / max(1, n_batches):.4f} "
                f"| val_auc {val_auc:.4f} | val_auc_pr {val_scores['auc_pr']:.4f}"
            )

        if np.isnan(val_auc):
            # Cannot judge improvement (e.g. tiny val set) -> keep going a bit.
            val_auc = best_auc

        if val_auc > best_auc + 1e-5:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.early_stopping_patience:
                if verbose:
                    print(f"early stopping at epoch {epoch} (best val_auc {best_auc:.4f})")
                break

    model.load_state_dict(best_state)
    return model
