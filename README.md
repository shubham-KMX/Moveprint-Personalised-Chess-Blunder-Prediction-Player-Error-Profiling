# Moveprint — Personalised Chess Blunder Prediction

Implementation of **"Blunder prediction in chess"** (Y. Rokach & B. Shapira,
*Applied Intelligence* 2026, https://doi.org/10.1007/s10489-026-07131-2).

This repository reproduces the paper's *unified, personalised, collaborative*
blunder-prediction model. Unlike the prior "one-model-per-player" paradigm
(McIlroy-Young et al. 2022), a **single** DeepFM-inspired network learns a shared
user-embedding space so that weaknesses generalise across players and to new
users.

## What is implemented

The winning **Architecture 14** from the paper (Table 1), which reaches
**0.801 AUC**:

- **Board embedding** — a *frozen* pre-trained CNN in the style of the
  RISEv2-mobile architecture from the CrazyAra engine (Czech et al. 2020).
  It preserves spatial dimensions (no pooling), uses MobileNetV2-style inverted
  residual blocks with Squeeze-and-Excitation layers, and the pyramid channel
  design (128 → +64/block → 896). The output of the layer *before* the penultimate
  layer is used as the board feature.
- **Two dedicated user embeddings** of size 25 (one for the deep component, one
  for the factorization inner product).
- **Deep component** — board features (500-d) + first user embedding + user rating
  + opponent rating + move ply → FC network → vector `x`.
- **Factorization component** — board projected to 25-d, inner product with the
  second user embedding → vector `y`.
- **Joint head** — concat(`x`, `y`) → FC layers → sigmoid → blunder probability.
- **Loss** — weighted binary cross-entropy (weight `0.08` on non-blunder moves),
  Adam, lr `1e-4`.

Also implemented:

- **Blunder labelling** — a move is a blunder if the win-probability drop is
  ≥ 10% (paper / McIlroy-Young 2020 definition).
- **Immediate vs non-immediate** blunder classification and the two specialised
  per-type models.
- **Chronological 80/20 split** per player (no leakage), and the `subset100` /
  `subset750` cohort construction described in Section 4.
- **Evaluation** — AUC and AUC-PR; per-Elo-group and per-game-phase analysis;
  the Elo-vs-user-id ablation (Table 2).

## Data

The paper uses a preprocessed Lichess dataset (McIlroy-Young et al. 2020) with
~3.9M moves. That corpus and the CrazyAra pre-trained weights are not bundled
here. Two data paths are provided:

1. **Real data** — point the pipeline at a directory of PGN files (optionally
   with an evaluation engine such as Stockfish available on `PATH`). See
   `moveprint/data/pgn_ingest.py`.
2. **Synthetic data** — a self-contained generator (`moveprint/data/synthetic.py`)
   produces a realistic, class-imbalanced dataset so the whole pipeline runs
   end-to-end offline. This is the default for the smoke test.

If CrazyAra weights are not present, the frozen CNN is randomly initialised and
frozen (still a valid fixed feature extractor); drop weights into `weights/` and
they will be loaded automatically.

## Quick start

```bash
pip install -r requirements.txt

# End-to-end smoke test on synthetic data (trains + evaluates)
python -m moveprint.run --synthetic --players 60 --epochs 3
```

## Layout

```
moveprint/
  config.py              # hyperparameters from the paper
  board.py               # board -> input planes encoding
  cnn.py                 # frozen RISEv2-mobile-style CNN board extractor
  model.py               # DeepFM-inspired Architecture 14
  losses.py              # weighted BCE
  train.py               # training loop + early stopping
  evaluate.py            # AUC / AUC-PR, per-phase, per-Elo, ablations
  data/
    schema.py            # Move / dataset record definitions
    labeling.py          # blunder + immediate/non-immediate labelling
    split.py             # chronological split, subset100 / subset750
    synthetic.py         # offline synthetic dataset
    pgn_ingest.py        # real PGN/Lichess ingestion
  run.py                 # CLI entrypoint
```

## Reference

Rokach, Y., Shapira, B. *Blunder prediction in chess.* Applied Intelligence 56, 92 (2026).
