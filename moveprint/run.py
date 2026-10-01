"""End-to-end entrypoint: build data -> train -> evaluate.

Examples
--------
Synthetic smoke test (offline, no external data):
    python -m moveprint.run --synthetic --players 60 --epochs 3

Real PGN data with Stockfish labelling:
    python -m moveprint.run --pgn-dir data/lichess --engine /usr/bin/stockfish
"""

from __future__ import annotations

import argparse
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import CNNConfig, DataConfig, ModelConfig, TrainConfig
from .data.schema import BlunderType, MoveRecord
from .data.split import chronological_split, select_subset100, select_subset750, filter_by_players
from .data.synthetic import generate_synthetic_dataset
from .dataset import MoveDataset, Normalizer, collate
from .evaluate import evaluate, evaluate_by_elo, evaluate_by_phase, predict, auc_scores
from .model import BlunderPredictor
from .train import train_model


def _stats(records: List[MoveRecord]) -> str:
    n = len(records)
    nb = sum(r.is_blunder for r in records)
    ni = sum(1 for r in records if r.blunder_type == BlunderType.IMMEDIATE)
    nn = sum(1 for r in records if r.blunder_type == BlunderType.NON_IMMEDIATE)
    return f"moves={n} blunders={nb} ({nb / max(1, n):.3f}) immediate={ni} non_immediate={nn}"


def load_records(args) -> List[MoveRecord]:
    if args.synthetic:
        print("[data] generating synthetic dataset ...")
        return generate_synthetic_dataset(
            num_players=args.players,
            games_per_player=args.games,
            moves_per_game=args.moves,
            seed=args.seed,
        )
    if args.csv:
        from .data.csv_ingest import ingest_moves_csv

        print(f"[data] ingesting labelled moves CSV from {args.csv} ...")
        return ingest_moves_csv(args.csv, max_rows=args.max_rows)
    if args.pgn_dir:
        from .data.pgn_ingest import ingest_pgn_dir

        print(f"[data] ingesting PGNs from {args.pgn_dir} ...")
        return ingest_pgn_dir(args.pgn_dir, engine_path=args.engine, max_games=args.max_games)
    raise SystemExit("Provide --synthetic, --csv, or --pgn-dir.")


def build_model(num_users: int, use_elo: bool, use_user_id: bool = True,
                board_extractor=None) -> BlunderPredictor:
    model_cfg = ModelConfig(use_elo=use_elo, use_user_id=use_user_id)
    return BlunderPredictor(num_users=num_users, model_cfg=model_cfg,
                            board_extractor=board_extractor)


def remap_user_ids(records: List[MoveRecord]) -> int:
    """Densify user ids to [0, num_users) so embedding tables are compact."""
    unique = sorted({r.user_id for r in records})
    mapping = {old: new for new, old in enumerate(unique)}
    for r in records:
        r.user_id = mapping[r.user_id]
    return len(unique)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Chess blunder prediction (Rokach & Shapira 2026).")
    parser.add_argument("--synthetic", action="store_true", help="use the offline synthetic dataset")
    parser.add_argument("--csv", type=str, default=None,
                        help="path to a pre-labelled moves CSV (real data)")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="cap rows read from --csv (quick runs)")
    parser.add_argument("--pgn-dir", type=str, default=None, help="directory of PGN files")
    parser.add_argument("--engine", type=str, default=None, help="path to a UCI engine (Stockfish)")
    parser.add_argument("--max-games", type=int, default=None)

    parser.add_argument("--players", type=int, default=60, help="synthetic: number of players")
    parser.add_argument("--games", type=int, default=20, help="synthetic: games per player")
    parser.add_argument("--moves", type=int, default=30, help="synthetic: moves per game")

    parser.add_argument("--subset", choices=["all", "100", "750"], default="all")
    parser.add_argument("--epochs", type=int, default=None, help="override max epochs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--ablation", action="store_true",
                        help="run the Elo vs user-id ablation (Table 2)")
    parser.add_argument("--per-type", action="store_true",
                        help="train specialised immediate / non-immediate models")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}")

    records = load_records(args)
    print(f"[data] full: {_stats(records)}")

    # Cohort selection.
    if args.subset == "100":
        keep = select_subset100(records, DataConfig().subset100_size)
        records = filter_by_players(records, keep)
        print(f"[data] subset100 -> {len(keep)} players")
    elif args.subset == "750":
        keep = select_subset750(records, DataConfig().subset750_size, seed=args.seed)
        records = filter_by_players(records, keep)
        print(f"[data] subset750 -> {len(keep)} players")

    num_users = remap_user_ids(records)
    print(f"[data] users={num_users}")

    # Chronological split.
    train_records, test_records = chronological_split(records, DataConfig().test_fraction)
    print(f"[data] train: {_stats(train_records)}")
    print(f"[data] test:  {_stats(test_records)}")

    normalizer = Normalizer.fit(train_records)

    train_cfg = TrainConfig(seed=args.seed)
    if args.epochs is not None:
        train_cfg.max_epochs = args.epochs
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size

    # Shared frozen board extractor (built once, reused across models).
    shared_extractor = None

    def make_test_loader(target_type=None):
        ds = MoveDataset(test_records, normalizer, target_type=target_type)
        return DataLoader(ds, batch_size=train_cfg.batch_size, shuffle=False, collate_fn=collate)

    # --- Main model (Architecture 14: user id only, no Elo) ----------------
    print("\n=== Training main blunder prediction model (Architecture 14) ===")
    model = build_model(num_users, use_elo=False, board_extractor=shared_extractor)
    shared_extractor = model.board_extractor
    model = train_model(model, train_records, normalizer, train_cfg, device)

    test_loader = make_test_loader()
    scores = evaluate(model, test_loader, device)
    print(f"[result] TEST  AUC={scores['auc']:.4f}  AUC-PR={scores['auc_pr']:.4f}")

    # Per-phase breakdown (Section 5.6).
    print("\n[analysis] by game phase:")
    for phase, s in evaluate_by_phase(model, test_loader, device).items():
        print(f"  {phase:8s} AUC={s['auc']:.4f} AUC-PR={s['auc_pr']:.4f} "
              f"n={s['sample_size']} blunders={s['blunder_count']}")

    # Per-Elo-group breakdown (Section 5.5).
    print("\n[analysis] by Elo group:")
    probs, targets, _ = predict(model, test_loader, device)
    for g in evaluate_by_elo(test_records, probs, targets, DataConfig().num_bins):
        print(f"  group {g['group']} [{g['rating_min']:.0f}-{g['rating_max']:.0f}] "
              f"AUC={g['auc']:.4f} AUC-PR={g['auc_pr']:.4f} n={g['sample_size']}")

    # --- Ablation: Elo vs user id (Table 2) --------------------------------
    if args.ablation:
        print("\n=== Ablation study (Table 2) ===")
        variants = {
            "elo+id": dict(use_elo=True, use_user_id=True),
            "id only": dict(use_elo=False, use_user_id=True),
            "elo only": dict(use_elo=True, use_user_id=False),
            "no user info": dict(use_elo=False, use_user_id=False),
        }
        for name, kw in variants.items():
            m = build_model(num_users, board_extractor=shared_extractor, **kw)
            m = train_model(m, train_records, normalizer, train_cfg, device,
                            verbose=False, board_embeddings=train_emb)
            s = evaluate(m, make_test_loader(), device)
            print(f"  {name:14s} AUC={s['auc']:.4f} AUC-PR={s['auc_pr']:.4f}")

    # --- Per-type specialised models (Section 5.4.2) -----------------------
    if args.per_type:
        print("\n=== Specialised blunder-type models (Section 5.4.2) ===")
        has_imm = any(r.blunder_type == BlunderType.IMMEDIATE for r in train_records)
        has_non = any(r.blunder_type == BlunderType.NON_IMMEDIATE for r in train_records)
        if not (has_imm and has_non):
            print("  [skip] dataset lacks fine-grained immediate / non_immediate "
                  "labels; per-type models need both.")
        else:
            for name, btype in (("immediate", BlunderType.IMMEDIATE),
                                ("non_immediate", BlunderType.NON_IMMEDIATE)):
                m = build_model(num_users, use_elo=False, board_extractor=shared_extractor)
                m = train_model(m, train_records, normalizer, train_cfg, device,
                                target_type=btype, verbose=False, board_embeddings=train_emb)
                s = evaluate(m, make_test_loader(target_type=btype), device)
                print(f"  {name:14s} AUC={s['auc']:.4f} AUC-PR={s['auc_pr']:.4f}")

    print("\n[done]")


if __name__ == "__main__":
    main()
