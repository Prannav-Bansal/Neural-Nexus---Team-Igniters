from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.data_loader import DisasterDataset, load_split_csv
from src.model import ModelConfig, MultiTaskEfficientNet
from src.utils import compute_metrics, ensure_dir, plot_confusion_matrix, save_json


def evaluate(args: argparse.Namespace) -> dict:
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    class_to_idx = checkpoint["class_to_idx"]
    severity_to_idx = checkpoint["severity_to_idx"]
    idx_to_class = {index: label for label, index in class_to_idx.items()}
    idx_to_severity = {index: label for label, index in severity_to_idx.items()}

    model = MultiTaskEfficientNet(
        ModelConfig(num_classes=len(class_to_idx), num_severity_levels=len(severity_to_idx))
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_df = load_split_csv(Path(args.data_dir) / "test.csv")
    dataset = DisasterDataset(test_df, class_to_idx, severity_to_idx, args.image_size, split="test")
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    true_class, pred_class = [], []
    true_severity, pred_severity = [], []

    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["image"].to(args.device))
            pred_class.extend(outputs["class_logits"].argmax(dim=1).cpu().tolist())
            pred_severity.extend(outputs["severity_logits"].argmax(dim=1).cpu().tolist())
            true_class.extend(batch["class_label"].tolist())
            true_severity.extend(batch["severity_label"].tolist())

    disaster_metrics = compute_metrics(true_class, pred_class)
    severity_metrics = compute_metrics(true_severity, pred_severity)

    disaster_report = classification_report(
        true_class,
        pred_class,
        target_names=[idx_to_class[i] for i in range(len(idx_to_class))],
        output_dict=True,
        zero_division=0,
    )
    severity_report = classification_report(
        true_severity,
        pred_severity,
        target_names=[idx_to_severity[i] for i in range(len(idx_to_severity))],
        output_dict=True,
        zero_division=0,
    )

    output_dir = ensure_dir(args.outputs_dir)
    plots_dir = ensure_dir(output_dir / "plots")
    logs_dir = ensure_dir(output_dir / "logs")

    disaster_cm = confusion_matrix(true_class, pred_class)
    severity_cm = confusion_matrix(true_severity, pred_severity)
    plot_confusion_matrix(
        disaster_cm,
        [idx_to_class[i] for i in range(len(idx_to_class))],
        "Disaster Type Confusion Matrix",
        plots_dir / "confusion_matrix_disaster.png",
    )
    plot_confusion_matrix(
        severity_cm,
        [idx_to_severity[i] for i in range(len(idx_to_severity))],
        "Severity Confusion Matrix",
        plots_dir / "confusion_matrix_severity.png",
    )

    results = {
        "disaster_metrics": disaster_metrics,
        "severity_metrics": severity_metrics,
        "disaster_report": disaster_report,
        "severity_report": severity_report,
        "test_samples": len(test_df),
    }
    save_json(results, logs_dir / "test_metrics.json")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the best model on the test split.")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pt")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--outputs-dir", type=str, default="outputs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
