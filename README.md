# Neural Nexus Disaster Management

Production-style disaster image analysis pipeline built with PyTorch and transfer learning. The system classifies disaster type, predicts a severity level, evaluates on a held-out test split, and generates Grad-CAM overlays for interpretability.

## What the project does

- Scans the local disaster image dataset automatically
- Detects folder-based labels and class imbalance
- Removes corrupted images during manifest creation
- Splits data into train/validation/test with stratification
- Trains an EfficientNet-B0 transfer-learning model with an SE attention block
- Predicts both disaster type and severity level
- Handles class imbalance with weighted losses
- Logs accuracy, precision, recall, and F1 per epoch
- Saves confusion matrices, reports, Grad-CAM outputs, and JSON predictions
- Exposes a simple Streamlit UI for image upload inference

## Dataset summary

The dataset was inferred from `Comprehensive Disaster Dataset(CDD)/CDD_Augmented` and uses folder-based labels.

- Classes: `Damaged_Infrastructure`, `Fire_Disaster`, `Human_Damage`, `Land_Disaster`, `Non_Damage`, `Water_Disaster`
- Total images: `18,303`
- Corrupted images removed: `0`
- Split sizes: train `12,812`, validation `2,745`, test `2,746`
- Class imbalance is strong, especially because `Non_Damage` dominates the dataset

Severity labels are derived from class semantics because the dataset does not provide separate severity annotations:

- `high`: `Damaged_Infrastructure`, `Fire_Disaster`, `Human_Damage`
- `medium`: `Land_Disaster`, `Water_Disaster`
- `low`: `Non_Damage`

## Project structure

```text
Neural Nexus Disaster Management/
├── data/
├── models/
├── outputs/
│   ├── logs/
│   ├── plots/
│   ├── gradcam/
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── utils.py
├── app.py
├── requirements.txt
├── README.md
```

## Setup

```bash
python -m pip install -r requirements.txt
```

## One-command pipeline

This command runs training, evaluation, and a 3-image inference demo:

```bash
python -m src.run_all --epochs 2 --batch-size 96 --image-size 160 --freeze-backbone
```

## Train only

```bash
python -m src.train --epochs 2 --batch-size 96 --image-size 160 --freeze-backbone
```

## Evaluate only

```bash
python -m src.evaluate --checkpoint models/best_model.pt
```

## Run inference on one image

```bash
python -m src.inference --image-path "Comprehensive Disaster Dataset(CDD)/CDD_Augmented/Water_Disaster/15000.jpg"
```

If no image path is supplied, the inference module picks a random image from the test split.

## Run the UI

```bash
python -m streamlit run app.py
```

## Latest run metrics

Held-out test set results from the final CPU run:

- Disaster type accuracy: `0.7433`
- Disaster type macro precision: `0.5732`
- Disaster type macro recall: `0.4507`
- Disaster type macro F1: `0.4739`
- Severity accuracy: `0.7535`
- Severity macro F1: `0.5988`

## Saved outputs

- Dataset summary: `data/dataset_summary.json`
- Train/val/test manifests: `data/train.csv`, `data/val.csv`, `data/test.csv`
- Best checkpoint: `models/best_model.pt`
- Training history: `outputs/logs/training_history.csv`
- Test metrics: `outputs/logs/test_metrics.json`
- Demo predictions: `outputs/logs/demo_predictions.json`
- Confusion matrices: `outputs/plots/`
- Grad-CAM overlays: `outputs/gradcam/`

## Notes

- Training ran on CPU in this environment, so the default command uses a frozen backbone and smaller image resolution to keep the workflow practical.
- The code will use GPU automatically if CUDA is available.
