"""Central configuration mirroring the hyperparameters reported in the paper.

Reference: Rokach & Shapira, "Blunder prediction in chess", Applied
Intelligence (2026). Values here match Section 3 (Methodology) and the winning
Architecture 14 in Table 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# --- Board encoding ---------------------------------------------------------
BOARD_SIZE = 8
# Input planes for the board tensor fed to the frozen CNN.
# 12 piece planes (6 white + 6 black) + side-to-move + castling(4) + en-passant
# + halfmove-clock + fullmove-number scaled  = 20 planes.
NUM_INPUT_PLANES = 20


# --- Blunder labelling ------------------------------------------------------
# A move is a blunder if the win probability drops by at least this amount.
# (McIlroy-Young 2020 / paper definition.)
BLUNDER_WINPROB_DROP = 0.10
# Material threshold (in pawns) used when disambiguating immediate blunders.
MATERIAL_LOSS_THRESHOLD = 1.0


@dataclass
class CNNConfig:
    """RISEv2-mobile-style frozen board feature extractor (CrazyAra).

    A 40-layer structure: an initial 3x3 conv (256 ch) followed by 13 inverted
    residual blocks. Pyramid design: 128 channels in the first block, +64 per
    block, reaching 896 in the final block. No pooling (spatial dims preserved).
    SE layers inside each block. The output of the layer *before* the penultimate
    layer is used as the board embedding.
    """

    in_planes: int = NUM_INPUT_PLANES
    stem_channels: int = 256
    num_blocks: int = 13
    start_channels: int = 128
    channel_step: int = 64
    se_ratio: int = 16
    # Dimension the extracted board feature map is flattened+projected to before
    # it enters the DeepFM model (component (b) output).
    embedding_dim: int = 512
    weights_path: str = "weights/crazyara_risev2.pt"


@dataclass
class ModelConfig:
    """DeepFM-inspired Architecture 14."""

    board_embedding_dim: int = 512          # from CNNConfig.embedding_dim
    board_fc_units: int = 500               # component (c)
    user_embedding_dim: int = 25            # component (d) — two of these
    fm_board_units: int = 25                # component (f) board projection
    # Deep component (e): two layers of 64 (paper: 64 neurons, ReLU).
    deep_units: List[int] = field(default_factory=lambda: [64, 64])
    # Joint head (g): four layers of 256, then 1-unit sigmoid.
    head_units: List[int] = field(default_factory=lambda: [256, 256, 256, 256])
    use_elo: bool = False          # Architecture 14 / best model: user id only
    use_user_id: bool = True


@dataclass
class TrainConfig:
    learning_rate: float = 1e-4            # Adam
    # Weighted BCE: weight assigned to the loss contribution of non-blunders.
    non_blunder_weight: float = 0.08
    batch_size: int = 256
    max_epochs: int = 50
    early_stopping_patience: int = 3       # on validation AUC
    val_fraction: float = 0.1              # split off the training set
    seed: int = 42


@dataclass
class DataConfig:
    # Chronological hold-out: most recent 20% of each player's games -> test.
    test_fraction: float = 0.20
    subset100_size: int = 100
    subset750_size: int = 750
    num_bins: int = 5                      # equal-frequency binning
