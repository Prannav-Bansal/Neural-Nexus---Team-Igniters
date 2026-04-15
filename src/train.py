from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_loader import (
    DisasterDataset,
    build_mappings,
    discover_dataset,
    load_split_csv,
    split_dataframe,
)
from src.model import ModelConfig, MultiTaskEfficientNet
from src.utils import compute_metrics, count_parameters, ensure_dir, get_device, save_json, set_seed


def build_data_loaders(
    data_dir: Path,
    raw_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[Dict[str, DataLoader], Dict[str, int], Dict[str, int], pd.DataFrame]:
    manifest_path = data_dir / "dataset_summary.json"
    dataframe, summary = discover_dataset(raw_dir, manifest_path)
    save_json(summary, manifest_path)

    split_paths = [data_dir / f"{split}.csv" for split in ("train", "val", "test")]
    if not all(path.exists() for path in split_paths):
        split_dataframe(dataframe, output_dir=data_dir)

    train_df = load_split_csv(data_dir / "train.csv")
    val_df = load_split_csv(data_dir / "val.csv")
    test_df = load_split_csv(data_dir / "test.csv")
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    class_to_idx, severity_to_idx = build_mappings(combined_df)

    datasets = {
        "train": DisasterDataset(train_df, class_to_idx, severity_to_idx, image_size, split="train"),
        "val": DisasterDataset(val_df, class_to_idx, severity_to_idx, image_size, split="val"),
        "test": DisasterDataset(test_df, class_to_idx, severity_to_idx, image_size, split="test"),
    }

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }
    return loaders, class_to_idx, severity_to_idx, combined_df


def build_losses(
    train_df: pd.DataFrame,
    class_to_idx: Dict[str, int],
    severity_to_idx: Dict[str, int],
    device: torch.device,
):
    class_counts = train_df["label"].value_counts()
    class_weights = []
    for class_name, _ in sorted(class_to_idx.items(), key=lambda item: item[1]):
        raw_weight = len(train_df) / (len(class_to_idx) * class_counts[class_name])
        class_weights.append(raw_weight ** 0.5)
    severity_counts = train_df["severity"].value_counts()
    severity_weights = []
    for severity_name, _ in sorted(severity_to_idx.items(), key=lambda item: item[1]):
        raw_weight = len(train_df) / (len(severity_to_idx) * severity_counts[severity_name])
        severity_weights.append(raw_weight ** 0.5)

    disaster_loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    severity_loss = nn.CrossEntropyLoss(weight=torch.tensor(severity_weights, dtype=torch.float32, device=device))
    return disaster_loss, severity_loss


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    disaster_loss_fn: nn.Module,
    severity_loss_fn: nn.Module,
) -> Dict[str, object]:
    is_train = optimizer is not None
    model.train(is_train)
    losses = []
    true_class, pred_class = [], []
    true_severity, pred_severity = [], []

    progress = tqdm(loader, leave=False)
    for batch in progress:
        images = batch["image"].to(device)
        class_targets = batch["class_label"].to(device)
        severity_targets = batch["severity_label"].to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            class_loss = disaster_loss_fn(outputs["class_logits"], class_targets)
            severity_loss = severity_loss_fn(outputs["severity_logits"], severity_targets)
            loss = class_loss + 0.35 * severity_loss

            if is_train:
                loss.backward()
                optimizer.step()

        losses.append(loss.item())
        pred_class.extend(outputs["class_logits"].argmax(dim=1).detach().cpu().tolist())
        true_class.extend(class_targets.detach().cpu().tolist())
        pred_severity.extend(outputs["severity_logits"].argmax(dim=1).detach().cpu().tolist())
        true_severity.extend(severity_targets.detach().cpu().tolist())

        progress.set_description(f"{'train' if is_train else 'eval '} loss={np.mean(losses):.4f}")

    disaster_metrics = compute_metrics(true_class, pred_class)
    severity_metrics = compute_metrics(true_severity, pred_severity)
    return {
        "loss": float(np.mean(losses)),
        "disaster": disaster_metrics,
        "severity": severity_metrics,
        "class_true": true_class,
        "class_pred": pred_class,
        "severity_true": true_severity,
        "severity_pred": pred_severity,
    }


def train(args: argparse.Namespace) -> Dict[str, object]:
    set_seed(args.seed)
    device = get_device()
    data_dir = ensure_dir(args.data_dir)
    ensure_dir(args.model_dir)
    ensure_dir(args.outputs_dir)
    ensure_dir(Path(args.outputs_dir) / "logs")

    loaders, class_to_idx, severity_to_idx, _ = build_data_loaders(
        data_dir=Path(args.data_dir),
        raw_dir=Path(args.raw_data_dir),
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    train_df = load_split_csv(Path(args.data_dir) / "train.csv")
    disaster_loss_fn, severity_loss_fn = build_losses(train_df, class_to_idx, severity_to_idx, device)

    model = MultiTaskEfficientNet(
        ModelConfig(
            num_classes=len(class_to_idx),
            num_severity_levels=len(severity_to_idx),
            dropout=args.dropout,
            freeze_backbone=args.freeze_backbone,
        )
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_metric = -1.0
    best_state = None
    patience_counter = 0
    history = []
    idx_to_class = {index: label for label, index in class_to_idx.items()}
    idx_to_severity = {index: label for label, index in severity_to_idx.items()}

    print(f"Device: {device}")
    print(f"Trainable params: {count_parameters(model):,}")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_metrics = run_epoch(model, loaders["train"], device, optimizer, disaster_loss_fn, severity_loss_fn)
        val_metrics = run_epoch(model, loaders["val"], device, None, disaster_loss_fn, severity_loss_fn)
        scheduler.step(val_metrics["disaster"]["f1"])

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_accuracy": train_metrics["disaster"]["accuracy"],
            "val_accuracy": val_metrics["disaster"]["accuracy"],
            "train_precision": train_metrics["disaster"]["precision"],
            "val_precision": val_metrics["disaster"]["precision"],
            "train_recall": train_metrics["disaster"]["recall"],
            "val_recall": val_metrics["disaster"]["recall"],
            "train_f1": train_metrics["disaster"]["f1"],
            "val_f1": val_metrics["disaster"]["f1"],
            "severity_val_f1": val_metrics["severity"]["f1"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": round(time.time() - epoch_start, 2),
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"train_acc={row['train_accuracy']:.4f} val_acc={row['val_accuracy']:.4f} "
            f"train_f1={row['train_f1']:.4f} val_f1={row['val_f1']:.4f}"
        )

        if val_metrics["disaster"]["f1"] > best_metric:
            best_metric = val_metrics["disaster"]["f1"]
            patience_counter = 0
            best_state = {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "class_to_idx": class_to_idx,
                "severity_to_idx": severity_to_idx,
                "config": vars(args),
                "idx_to_class": idx_to_class,
                "idx_to_severity": idx_to_severity,
                "best_val_metrics": val_metrics,
            }
            torch.save(best_state, Path(args.model_dir) / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    history_df = pd.DataFrame(history)
    history_df.to_csv(Path(args.outputs_dir) / "logs" / "training_history.csv", index=False)
    save_json(history, Path(args.outputs_dir) / "logs" / "training_history.json")

    if best_state is None:
        raise RuntimeError("Training ended without a best checkpoint.")

    report = classification_report(
        val_metrics["class_true"],
        val_metrics["class_pred"],
        target_names=[idx_to_class[i] for i in range(len(idx_to_class))],
        output_dict=True,
        zero_division=0,
    )
    save_json(report, Path(args.outputs_dir) / "logs" / "validation_classification_report.json")

    summary = {
        "device": str(device),
        "num_classes": len(class_to_idx),
        "num_severity_levels": len(severity_to_idx),
        "class_names": [idx_to_class[i] for i in range(len(idx_to_class))],
        "severity_names": [idx_to_severity[i] for i in range(len(idx_to_severity))],
        "best_val_f1": best_metric,
        "best_val_accuracy": best_state["best_val_metrics"]["disaster"]["accuracy"],
        "train_samples": int(len(loaders["train"].dataset)),
        "val_samples": int(len(loaders["val"].dataset)),
        "test_samples": int(len(loaders["test"].dataset)),
    }
    save_json(summary, Path(args.outputs_dir) / "logs" / "training_summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the disaster management model.")
    parser.add_argument("--raw-data-dir", type=str, default="Comprehensive Disaster Dataset(CDD)/CDD_Augmented")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--outputs-dir", type=str, default="outputs")
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
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
