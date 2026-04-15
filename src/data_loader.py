from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils import ensure_dir, infer_severity, save_json


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DisasterDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        class_to_idx: Dict[str, int],
        severity_to_idx: Dict[str, int],
        image_size: int,
        split: str,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.severity_to_idx = severity_to_idx
        self.transform = build_transforms(image_size=image_size, split=split)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        row = self.dataframe.iloc[index]
        image_path = Path(row["filepath"])
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        tensor = self.transform(image)
        return {
            "image": tensor,
            "class_label": torch.tensor(self.class_to_idx[row["label"]], dtype=torch.long),
            "severity_label": torch.tensor(self.severity_to_idx[row["severity"]], dtype=torch.long),
            "path": str(image_path),
        }


def build_transforms(image_size: int, split: str) -> transforms.Compose:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize((image_size + 16, image_size + 16)),
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.15),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.25, contrast=0.2, saturation=0.2, hue=0.03),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2
                ),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x + 0.03 * torch.randn_like(x)).clamp(0, 1)),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


def discover_dataset(raw_root: str | Path, summary_path: str | Path | None = None) -> Tuple[pd.DataFrame, Dict[str, object]]:
    raw_root = Path(raw_root)
    rows: List[Dict[str, str]] = []
    corrupt_files: List[str] = []
    image_modes = Counter()
    width_heights: List[Tuple[int, int]] = []

    class_dirs = sorted([path for path in raw_root.iterdir() if path.is_dir()])
    for class_dir in class_dirs:
        for file_path in class_dir.rglob("*"):
            if not file_path.is_file() or file_path.suffix.lower() not in VALID_EXTENSIONS:
                continue
            try:
                with Image.open(file_path) as image:
                    image.verify()
                with Image.open(file_path) as image:
                    image_modes[image.mode] += 1
                    width_heights.append(image.size)
            except (UnidentifiedImageError, OSError, ValueError):
                corrupt_files.append(str(file_path))
                continue

            rows.append(
                {
                    "filepath": str(file_path),
                    "label": class_dir.name,
                    "severity": infer_severity(class_dir.name),
                }
            )

    dataframe = pd.DataFrame(rows)
    label_counts = dataframe["label"].value_counts().sort_index().to_dict()
    severity_counts = dataframe["severity"].value_counts().sort_index().to_dict()

    width_avg = round(sum(width for width, _ in width_heights) / len(width_heights), 2)
    height_avg = round(sum(height for _, height in width_heights) / len(width_heights), 2)
    imbalance_ratio = round(max(label_counts.values()) / min(label_counts.values()), 2)

    summary = {
        "raw_root": str(raw_root),
        "label_source": "folder_based",
        "num_classes": len(label_counts),
        "classes": sorted(label_counts.keys()),
        "class_distribution": label_counts,
        "severity_distribution": severity_counts,
        "total_images": int(len(dataframe)),
        "corrupt_images_removed": len(corrupt_files),
        "sample_corrupt_images": corrupt_files[:10],
        "image_modes": dict(image_modes),
        "avg_width": width_avg,
        "avg_height": height_avg,
        "imbalance_ratio": imbalance_ratio,
    }

    if summary_path is not None:
        save_json(summary, summary_path)

    return dataframe, summary


def split_dataframe(
    dataframe: pd.DataFrame,
    output_dir: str | Path,
    random_state: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Dict[str, pd.DataFrame]:
    output_dir = ensure_dir(output_dir)
    train_df, temp_df = train_test_split(
        dataframe,
        test_size=(1.0 - train_ratio),
        stratify=dataframe["label"],
        random_state=random_state,
    )
    relative_val_ratio = val_ratio / (1.0 - train_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - relative_val_ratio),
        stratify=temp_df["label"],
        random_state=random_state,
    )

    splits = {
        "train": train_df.sort_values("filepath").reset_index(drop=True),
        "val": val_df.sort_values("filepath").reset_index(drop=True),
        "test": test_df.sort_values("filepath").reset_index(drop=True),
    }

    for split_name, split_df in splits.items():
        split_df.to_csv(output_dir / f"{split_name}.csv", index=False)

    return splits


def load_split_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_mappings(dataframe: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
    classes = sorted(dataframe["label"].unique())
    severities = sorted(dataframe["severity"].unique())
    class_to_idx = {label: index for index, label in enumerate(classes)}
    severity_to_idx = {label: index for index, label in enumerate(severities)}
    return class_to_idx, severity_to_idx
