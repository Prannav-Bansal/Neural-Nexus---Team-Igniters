from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        weights = self.pool(x).view(batch, channels)
        weights = self.fc(weights).view(batch, channels, 1, 1)
        return x * weights


@dataclass
class ModelConfig:
    num_classes: int
    num_severity_levels: int
    dropout: float = 0.35
    freeze_backbone: bool = False


class MultiTaskEfficientNet(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        try:
            backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        except Exception:
            backbone = efficientnet_b0(weights=None)

        self.features = backbone.features
        self.attention = SEBlock(1280)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.shared_head = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
        )
        self.classifier = nn.Linear(512, config.num_classes)
        self.severity_head = nn.Linear(512, config.num_severity_levels)

        if config.freeze_backbone:
            for parameter in self.features.parameters():
                parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.features(x)
        x = self.attention(x)
        x = self.pool(x).flatten(1)
        shared = self.shared_head(x)
        return {
            "class_logits": self.classifier(shared),
            "severity_logits": self.severity_head(shared),
            "features": shared,
        }
