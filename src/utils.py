import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


SEVERITY_MAP = {
    "Non_Damage": "low",
    "Human_Damage": "high",
    "Damaged_Infrastructure": "high",
    "Fire_Disaster": "high",
    "Water_Disaster": "medium",
    "Land_Disaster": "medium",
}


def ensure_dir(path: os.PathLike[str] | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(payload: Dict[str, Any], path: os.PathLike[str] | str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path: os.PathLike[str] | str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def compute_metrics(y_true: Iterable[int], y_pred: Iterable[int], average: str = "macro") -> Dict[str, float]:
    y_true = list(y_true)
    y_pred = list(y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: List[str],
    title: str,
    path: os.PathLike[str] | str,
) -> None:
    ensure_dir(Path(path).parent)
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="YlOrRd", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def pretty_class_name(label: str) -> str:
    return label.replace("_", " ")


def infer_severity(label: str) -> str:
    return SEVERITY_MAP.get(label, "medium")
