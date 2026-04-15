from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from src.data_loader import VALID_EXTENSIONS, build_transforms, load_split_csv
from src.model import ModelConfig, MultiTaskEfficientNet
from src.utils import ensure_dir, save_json


class GradCAM:
    def __init__(self, model: MultiTaskEfficientNet, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, image_tensor: torch.Tensor, class_index: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(image_tensor)
        score = outputs["class_logits"][:, class_index].sum()
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def load_model(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[MultiTaskEfficientNet, Dict[str, int], Dict[str, int]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_to_idx = checkpoint["class_to_idx"]
    severity_to_idx = checkpoint["severity_to_idx"]
    model = MultiTaskEfficientNet(
        ModelConfig(num_classes=len(class_to_idx), num_severity_levels=len(severity_to_idx))
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_to_idx, severity_to_idx


def overlay_gradcam(image: Image.Image, cam: np.ndarray, save_path: str | Path) -> None:
    image_np = np.array(image.resize((cam.shape[1], cam.shape[0]))).astype(np.float32) / 255.0
    heatmap = cm.get_cmap("jet")(cam)[..., :3]
    blended = np.clip(0.45 * heatmap + 0.55 * image_np, 0, 1)
    ensure_dir(Path(save_path).parent)
    plt.figure(figsize=(6, 6))
    plt.imshow(blended)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=220)
    plt.close()


def predict_image(
    image_path: str | Path,
    checkpoint_path: str | Path = "models/best_model.pt",
    outputs_dir: str | Path = "outputs",
    image_size: int = 224,
    device: str | torch.device | None = None,
) -> Dict[str, object]:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, class_to_idx, severity_to_idx = load_model(checkpoint_path, device)
    idx_to_class = {index: label for label, index in class_to_idx.items()}
    idx_to_severity = {index: label for label, index in severity_to_idx.items()}

    image_path = Path(image_path)
    transform = build_transforms(image_size, split="test")
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        outputs = model(input_tensor)

        class_probs = torch.softmax(outputs["class_logits"], dim=1)
        severity_probs = torch.softmax(outputs["severity_logits"], dim=1)
        class_index = int(class_probs.argmax(dim=1).item())
        severity_index = int(severity_probs.argmax(dim=1).item())
        confidence = float(class_probs.max(dim=1).values.item())

        gradcam = GradCAM(model, model.features[-1])
        cam = gradcam(input_tensor, class_index)
        gradcam_path = Path(outputs_dir) / "gradcam" / f"{image_path.stem}_gradcam.png"
        overlay_gradcam(image, cam, gradcam_path)
        gradcam.close()

    values, indices = torch.topk(class_probs.squeeze(0), k=min(3, len(class_to_idx)))
    result = {
        "image_path": str(image_path),
        "predicted_disaster_type": idx_to_class[class_index],
        "predicted_severity_level": idx_to_severity[severity_index],
        "confidence_score": round(confidence, 4),
        "top3_disaster_predictions": [
            {"label": idx_to_class[int(index)], "confidence": round(float(score), 4)}
            for score, index in zip(values.tolist(), indices.tolist())
        ],
        "gradcam_path": str(gradcam_path),
    }
    prediction_path = Path(outputs_dir) / "logs" / f"{image_path.stem}_prediction.json"
    save_json(result, prediction_path)
    return result


def demo_predictions(
    checkpoint_path: str | Path,
    data_dir: str | Path = "data",
    outputs_dir: str | Path = "outputs",
    sample_count: int = 3,
) -> List[Dict[str, object]]:
    test_df = load_split_csv(Path(data_dir) / "test.csv")
    sample_count = min(sample_count, len(test_df))
    selected = test_df.sample(n=sample_count, random_state=11)["filepath"].tolist()
    results = [predict_image(path, checkpoint_path=checkpoint_path, outputs_dir=outputs_dir) for path in selected]
    save_json({"demo_predictions": results}, Path(outputs_dir) / "logs" / "demo_predictions.json")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference and save prediction JSON.")
    parser.add_argument("--image-path", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pt")
    parser.add_argument("--outputs-dir", type=str, default="outputs")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--demo", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.demo:
        results = demo_predictions(args.checkpoint, data_dir=args.data_dir, outputs_dir=args.outputs_dir)
        for item in results:
            print(item)
    else:
        if not args.image_path:
            test_df = load_split_csv(Path(args.data_dir) / "test.csv")
            args.image_path = test_df.sample(n=1, random_state=17)["filepath"].iloc[0]
        print(predict_image(args.image_path, checkpoint_path=args.checkpoint, outputs_dir=args.outputs_dir))
