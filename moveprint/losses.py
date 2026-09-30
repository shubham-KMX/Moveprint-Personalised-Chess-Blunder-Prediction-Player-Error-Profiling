"""Weighted binary cross-entropy loss.

The paper addresses class imbalance with a weighted BCE that assigns a weight of
0.08 to the loss contribution from non-blunder moves (Section 3.1). Blunder
moves keep weight 1.0.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    non_blunder_weight: float = 0.08,
) -> torch.Tensor:
    """Weighted BCE with logits.

    Positive (blunder) examples get weight 1.0; negatives get
    `non_blunder_weight`.
    """
    weights = torch.where(
        targets > 0.5,
        torch.ones_like(targets),
        torch.full_like(targets, non_blunder_weight),
    )
    loss = F.binary_cross_entropy_with_logits(logits, targets, weight=weights)
    return loss
