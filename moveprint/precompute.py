"""Precompute frozen-CNN board embeddings.

Because the RISEv2-mobile board extractor is *frozen* (Section 3), its output
for a given board never changes during training. We therefore run it once over
every move and cache the resulting embedding. Training then operates on the
cached vectors, which is numerically identical to recomputing them each step but
dramatically faster — essential for the ~478k-move real dataset on CPU.

Embeddings are keyed by the FEN-equivalent board tensor so identical positions
share one forward pass.
"""

from __future__ import annotations

import hashlib
from typing import List, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

from .cnn import RISEv2Mobile
from .data.schema import MoveRecord


def _board_key(planes: np.ndarray) -> bytes:
    return hashlib.sha1(np.ascontiguousarray(planes).view(np.uint8)).digest()


@torch.no_grad()
def precompute_board_embeddings(
    records: Sequence[MoveRecord],
    extractor: RISEv2Mobile,
    device: Optional[torch.device] = None,
    batch_size: int = 256,
    dedupe: bool = True,
    show_progress: bool = True,
) -> np.ndarray:
    """Return an (N, D) float32 array of board embeddings aligned to `records`.

    When `dedupe` is True, identical board tensors are embedded once and reused.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = extractor.to(device).eval()

    n = len(records)
    emb_dim = extractor.cfg.embedding_dim
    out = np.zeros((n, emb_dim), dtype=np.float32)

    if dedupe:
        key_to_rows: dict[bytes, List[int]] = {}
        unique_keys: List[bytes] = []
        unique_planes: List[np.ndarray] = []
        for i, r in enumerate(records):
            k = _board_key(r.board_planes)
            if k not in key_to_rows:
                key_to_rows[k] = []
                unique_keys.append(k)
                unique_planes.append(r.board_planes)
            key_to_rows[k].append(i)

        unique_emb = _embed_planes(unique_planes, extractor, device, batch_size, show_progress)
        for k, emb in zip(unique_keys, unique_emb):
            for row in key_to_rows[k]:
                out[row] = emb
        if show_progress:
            print(f"[precompute] {n} moves -> {len(unique_keys)} unique boards embedded")
    else:
        planes = [r.board_planes for r in records]
        out[:] = _embed_planes(planes, extractor, device, batch_size, show_progress)

    return out


@torch.no_grad()
def _embed_planes(planes_list, extractor, device, batch_size, show_progress) -> np.ndarray:
    embs = []
    rng = range(0, len(planes_list), batch_size)
    iterator = tqdm(rng, desc="[precompute] embedding boards") if show_progress else rng
    for start in iterator:
        chunk = planes_list[start:start + batch_size]
        batch = torch.from_numpy(np.stack(chunk)).float().to(device)
        e = extractor(batch).cpu().numpy().astype(np.float32)
        embs.append(e)
    return np.concatenate(embs, axis=0)
