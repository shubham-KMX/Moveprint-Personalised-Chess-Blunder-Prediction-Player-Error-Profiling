"""Evaluation metrics and analyses.

Primary metrics (Section 5.1): AUC and AUC-PR (average precision). Also provides
per-game-phase and per-Elo-group breakdowns (Sections 5.5, 5.6).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from .data.schema import GamePhase


@torch.no_grad()
def predict(model, loader: DataLoader, device: torch.device):
    """Run the model over a loader, returning (probs, targets, phases)."""
    model.eval()
    probs: List[float] = []
    targets: List[float] = []
    phases: List[int] = []
    for batch in loader:
        logits = model(
            batch["board_planes"].to(device),
            batch["user_id"].to(device),
            batch["user_rating"].to(device),
            batch["opp_rating"].to(device),
            batch["ply"].to(device),
        )
        p = torch.sigmoid(logits).cpu().numpy()
        probs.extend(p.tolist())
        targets.extend(batch["target"].numpy().tolist())
        phases.extend(batch["phase"].numpy().tolist())
    return np.array(probs), np.array(targets), np.array(phases)


def auc_scores(probs: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """AUC and AUC-PR. Returns NaNs safely when a class is missing."""
    out = {"auc": float("nan"), "auc_pr": float("nan")}
    if len(np.unique(targets)) < 2:
        return out
    out["auc"] = float(roc_auc_score(targets, probs))
    out["auc_pr"] = float(average_precision_score(targets, probs))
    return out


def evaluate(model, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    probs, targets, _ = predict(model, loader, device)
    return auc_scores(probs, targets)


def evaluate_by_phase(model, loader: DataLoader, device: torch.device) -> Dict[str, Dict[str, float]]:
    """AUC / AUC-PR broken down by game phase (Section 5.6)."""
    probs, targets, phases = predict(model, loader, device)
    result: Dict[str, Dict[str, float]] = {}
    names = {int(GamePhase.OPENING): "opening",
             int(GamePhase.MIDDLE): "middle",
             int(GamePhase.ENDGAME): "endgame"}
    for phase_val, name in names.items():
        mask = phases == phase_val
        scores = auc_scores(probs[mask], targets[mask])
        scores["sample_size"] = int(mask.sum())
        scores["blunder_count"] = int(targets[mask].sum())
        result[name] = scores
    return result


def evaluate_by_elo(records, probs: np.ndarray, targets: np.ndarray,
                    num_groups: int = 5) -> List[Dict[str, float]]:
    """AUC / AUC-PR by equal-frequency Elo group (Section 5.5).

    `records` must align 1:1 with probs/targets order (i.e. produced from the
    same loader with shuffle disabled).
    """
    ratings = np.array([r.user_rating for r in records])
    order = ratings.argsort()
    ranks = order.argsort()
    groups = np.clip((ranks * num_groups / len(ratings)).astype(int), 0, num_groups - 1)

    out: List[Dict[str, float]] = []
    for g in range(num_groups):
        mask = groups == g
        scores = auc_scores(probs[mask], targets[mask])
        scores["group"] = g
        scores["rating_min"] = float(ratings[mask].min()) if mask.any() else float("nan")
        scores["rating_max"] = float(ratings[mask].max()) if mask.any() else float("nan")
        scores["sample_size"] = int(mask.sum())
        out.append(scores)
    return out
