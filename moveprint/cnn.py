"""Frozen RISEv2-mobile-style CNN board feature extractor (CrazyAra).

The paper uses the RISEv2-mobile architecture from the CrazyAra engine
(Czech et al. 2020) as a *frozen* board feature extractor. Key properties from
Section 3:

  * 40-layer deep structure.
  * Initial 3x3 conv with 256 channels.
  * 13 inverted residual blocks. Each block: 1x1 conv -> 3x3 (depthwise)
    conv -> 1x1 conv, with a Squeeze-and-Excitation layer.
  * "Pyramid" design: 128 channels in the first block, expanding by 64 channels
    per block up to 896 channels in the final layers.
  * No pooling / down-sampling -> spatial dimensions (8x8) are preserved.
  * The output of the layer *before* the penultimate layer is used as the board
    embedding.

If real CrazyAra weights are available they are loaded and the network is
frozen. Otherwise the network is randomly initialised and frozen, which is still
a valid fixed feature extractor for reproducing the pipeline offline.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn

from .config import CNNConfig


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, ratio: int = 16):
        super().__init__()
        hidden = max(1, channels // ratio)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.pool(x)
        s = self.act(self.fc1(s))
        s = self.gate(self.fc2(s))
        return x * s


class InvertedResidualBlock(nn.Module):
    """MobileNetV2-style inverted residual block with SE, no spatial reduction.

    1x1 expand -> 3x3 depthwise -> 1x1 project, plus SE. Padding preserves the
    8x8 spatial dimensions. A residual connection is used when in/out channels
    match.
    """

    def __init__(self, in_ch: int, out_ch: int, expand_ratio: int = 4, se_ratio: int = 16):
        super().__init__()
        hidden = in_ch * expand_ratio
        self.use_residual = in_ch == out_ch

        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.se = SqueezeExcitation(hidden, ratio=se_ratio)
        self.project = nn.Sequential(
            nn.Conv2d(hidden, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.se(out)
        out = self.project(out)
        if self.use_residual:
            out = out + x
        return out


class RISEv2Mobile(nn.Module):
    """Frozen board feature extractor.

    forward() returns the board embedding: the flattened feature map from the
    layer before the penultimate layer, projected to `cfg.embedding_dim`.
    """

    def __init__(self, cfg: CNNConfig):
        super().__init__()
        self.cfg = cfg

        # Stem: 3x3 conv, 256 channels, padding keeps 8x8.
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.in_planes, cfg.stem_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cfg.stem_channels),
            nn.ReLU(inplace=True),
        )

        # Pyramid of inverted residual blocks.
        blocks = []
        in_ch = cfg.stem_channels
        for i in range(cfg.num_blocks):
            out_ch = cfg.start_channels + i * cfg.channel_step  # 128, 192, ..., 896
            blocks.append(
                InvertedResidualBlock(
                    in_ch, out_ch,
                    expand_ratio=cfg.expand_ratio,
                    se_ratio=cfg.se_ratio,
                )
            )
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)

        # `penultimate` conv is the final represented layer; we tap the block
        # output *before* it (i.e. the last block output) as the embedding, per
        # the paper ("output of the layer prior to the penultimate layer").
        self.final_channels = in_ch  # 896

        # Projection of the final feature map to a compact board embedding. When
        # pooling, the input is `final_channels`; otherwise the full flattened
        # `final_channels * 8 * 8` map (paper-scale).
        self.pool_before_projection = cfg.pool_before_projection
        proj_in = self.final_channels if cfg.pool_before_projection else self.final_channels * 8 * 8
        self.project = nn.Linear(proj_in, cfg.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        feat = x
        for block in self.blocks:
            feat = block(feat)
        # feat: (B, final_channels, 8, 8) -- the pre-penultimate representation.
        if self.pool_before_projection:
            flat = feat.mean(dim=(2, 3))          # global average pool -> (B, C)
        else:
            flat = feat.flatten(start_dim=1)
        return self.project(flat)


def build_frozen_board_extractor(cfg: Optional[CNNConfig] = None,
                                 device: Optional[torch.device] = None) -> RISEv2Mobile:
    """Build the board extractor, load CrazyAra weights if present, and freeze it."""
    cfg = cfg or CNNConfig()
    model = RISEv2Mobile(cfg)

    loaded = False
    if cfg.weights_path and os.path.exists(cfg.weights_path):
        try:
            state = torch.load(cfg.weights_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
            loaded = True
        except Exception as exc:  # pragma: no cover
            print(f"[cnn] failed to load weights from {cfg.weights_path}: {exc}")

    if not loaded:
        print(
            "[cnn] CrazyAra pretrained weights not found; using a randomly "
            "initialised *frozen* extractor (fixed feature map)."
        )

    # Freeze: the CNN is a fixed feature extractor in the paper.
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    if device is not None:
        model = model.to(device)
    return model
