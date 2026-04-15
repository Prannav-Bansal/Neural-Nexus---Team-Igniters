from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.evaluate import evaluate
from src.inference import demo_predictions
from src.train import train


def run_all(args: argparse.Namespace) -> dict:
    train_summary = train(args)
    eval_summary = evaluate(args)
    demo_summary = demo_predictions(args.checkpoint, data_dir=args.data_dir, outputs_dir=args.outputs_dir, sample_count=3)
    final_summary = {
        "train_summary": train_summary,
        "evaluation": eval_summary,
        "demo_predictions": demo_summary,
    }
    Path(args.outputs_dir, "logs").mkdir(parents=True, exist_ok=True)
    with Path(args.outputs_dir, "logs", "run_all_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(final_summary, handle, indent=2)
    return final_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train, evaluate, and run demo inference.")
    parser.add_argument("--raw-data-dir", type=str, default="Comprehensive Disaster Dataset(CDD)/CDD_Augmented")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--outputs-dir", type=str, default="outputs")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pt")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run_all(parse_args())
