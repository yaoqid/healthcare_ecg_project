"""Train the LSTM risk model on synthetic sequences or PTB-XL histories."""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.lstm_model import get_model
from data.data_loader import generate_synthetic_sequences, get_lstm_dataloaders, load_ptbxl_sequence_dataframe


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Run one training epoch and return mean loss and argmax accuracy."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for seqs, labels in loader:
        seqs, labels = seqs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(seqs)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate the model and return loss, accuracy, labels, and positive-class scores."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_probs = [], []

    for seqs, labels in loader:
        seqs, labels = seqs.to(device), labels.to(device)
        outputs = model(seqs)
        loss = criterion(outputs, labels)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        total_loss += loss.item()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

    return total_loss / len(loader), correct / total, np.asarray(all_labels, dtype=np.int64), np.asarray(all_probs, dtype=np.float32)


def predict_from_probs(probs, threshold: float) -> np.ndarray:
    """Convert positive-class probabilities into hard predictions."""
    return (np.asarray(probs) >= threshold).astype(np.int64)


def compute_metrics(labels, probs, threshold: float):
    """Compute imbalance-aware metrics for binary risk prediction."""
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float32)
    preds = predict_from_probs(probs, threshold)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )

    metrics = {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "balanced_acc": float(balanced_accuracy_score(labels, preds)),
        "mcc": float(matthews_corrcoef(labels, preds)),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
        "pred_positive_rate": float(preds.mean()) if len(preds) else 0.0,
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        metrics["roc_auc"] = None
    try:
        metrics["pr_auc"] = float(average_precision_score(labels, probs))
    except ValueError:
        metrics["pr_auc"] = None
    return metrics


def describe_label_distribution(name: str, labels) -> dict:
    """Print and return class counts for the current split."""
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=2)
    total = counts.sum()
    distribution = {
        "negative_count": int(counts[0]),
        "positive_count": int(counts[1]),
        "positive_rate": float(counts[1] / total) if total else 0.0,
    }
    print(
        f"{name}: {distribution['negative_count']} negative / {distribution['positive_count']} positive "
        f"(positive rate {distribution['positive_rate']:.3f})"
    )
    return distribution


def build_class_weights(labels, device: torch.device) -> torch.Tensor:
    """Create inverse-frequency class weights for the training split."""
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=2).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def get_selection_score(metric_name: str, val_loss: float, val_acc: float, val_metrics: dict) -> float:
    """Return the scalar score used to decide which checkpoint is best."""
    if metric_name == "loss":
        return -val_loss
    if metric_name == "acc":
        return val_acc
    if metric_name == "f1":
        return val_metrics["f1"]
    if metric_name == "macro_f1":
        return val_metrics["macro_f1"]
    if metric_name == "balanced_acc":
        return val_metrics["balanced_acc"]
    if metric_name == "mcc":
        return val_metrics["mcc"]
    if metric_name == "roc_auc":
        return float("-inf") if val_metrics["roc_auc"] is None else val_metrics["roc_auc"]
    if metric_name == "pr_auc":
        return float("-inf") if val_metrics["pr_auc"] is None else val_metrics["pr_auc"]
    raise ValueError(f"Unsupported selection metric: {metric_name}")


def tune_threshold(labels, probs, metric_name: str) -> tuple[float, dict]:
    """Search for a validation threshold that improves hard predictions."""
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float32)
    candidate_thresholds = np.unique(np.concatenate((
        np.linspace(0.1, 0.9, 17, dtype=np.float32),
        probs,
    )))

    best_threshold = 0.5
    best_metrics = None
    best_score = float("-inf")
    for threshold in candidate_thresholds:
        metrics = compute_metrics(labels, probs, float(threshold))
        score = get_selection_score(metric_name, val_loss=0.0, val_acc=0.0, val_metrics=metrics)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def compute_feature_stats(sequence_dataset) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean and std from the training sequences only."""
    stacked = np.concatenate(sequence_dataset.sequences, axis=0).astype(np.float32)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_feature_standardization(sequence_dataset, mean: np.ndarray, std: np.ndarray):
    """Normalize sequences in-place with training-set statistics."""
    sequence_dataset.sequences = [
        ((seq - mean) / std).astype(np.float32)
        for seq in sequence_dataset.sequences
    ]


def set_seed(seed: int):
    """Seed NumPy and PyTorch for repeatable runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_best_device() -> torch.device:
    """Select the best available accelerator in priority: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(args):
    set_seed(args.seed)
    device = get_best_device()
    print(f"🖥️  Training on: {device}")
    os.makedirs("checkpoints", exist_ok=True)

    print("\n📦 Loading patient sequences...")
    if args.real_data:
        if not args.ptbxl_dir:
            raise ValueError("--ptbxl-dir is required when using --real-data.")
        df = load_ptbxl_sequence_dataframe(
            args.ptbxl_dir,
            sampling_rate=args.sampling_rate,
            lead_index=args.lead_index,
            limit=args.limit,
        )
        if df.empty:
            raise ValueError(
                "No multi-record PTB-XL patient sequences were created. "
                "Try increasing --limit or reducing --seq-len."
            )
        print(
            f"Loaded {len(df)} PTB-XL patient records with heuristic feature extraction "
            f"at {args.sampling_rate}Hz."
        )
    else:
        df = generate_synthetic_sequences(n_patients=args.n_patients, seq_len=args.seq_len + 6)

    train_loader, val_loader = get_lstm_dataloaders(df, batch_size=args.batch_size, seq_len=args.seq_len)
    feature_mean, feature_std = compute_feature_stats(train_loader.dataset)
    apply_feature_standardization(train_loader.dataset, feature_mean, feature_std)
    apply_feature_standardization(val_loader.dataset, feature_mean, feature_std)
    train_distribution = describe_label_distribution("Train sequence distribution", train_loader.dataset.labels)
    val_distribution = describe_label_distribution("Val sequence distribution", val_loader.dataset.labels)
    print("Applied train-set feature standardization to LSTM inputs.")

    print("\n🧠 Building LSTM model...")
    model = get_model(input_size=8, num_classes=2).to(device)

    class_weights = build_class_weights(train_loader.dataset.labels, device) if args.use_class_weights else None
    if class_weights is not None:
        print(f"Using class weights: negative={class_weights[0].item():.3f}, positive={class_weights[1].item():.3f}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\n🚀 Starting LSTM training for {args.epochs} epochs...\n")
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_macro_f1": [],
        "val_balanced_acc": [],
        "val_mcc": [],
        "val_roc_auc": [],
        "val_pr_auc": [],
        "val_threshold": [],
        "val_confusion_matrix": [],
        "val_pred_positive_rate": [],
    }
    best_score = float("-inf")
    best_metrics = {}
    print(f"Selecting best checkpoint by validation {args.selection_metric}.")
    print(f"Tuning the binary decision threshold by validation {args.threshold_metric}.")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, _, val_labels, val_probs = evaluate(model, val_loader, criterion, device)
        threshold, val_metrics = tune_threshold(val_labels, val_probs, args.threshold_metric)
        val_preds = predict_from_probs(val_probs, threshold)
        val_acc = float((val_preds == val_labels).mean())
        selection_score = get_selection_score(args.selection_metric, val_loss, val_acc, val_metrics)
        roc_auc_display = "N/A" if val_metrics["roc_auc"] is None else f"{val_metrics['roc_auc']:.4f}"
        pr_auc_display = "N/A" if val_metrics["pr_auc"] is None else f"{val_metrics['pr_auc']:.4f}"
        tn, fp = val_metrics["confusion_matrix"][0]
        fn, tp = val_metrics["confusion_matrix"][1]
        pred_positive_rate = val_metrics["pred_positive_rate"]
        scheduler.step()

        if selection_score > best_score:
            best_score = selection_score
            torch.save(model.state_dict(), "checkpoints/lstm_best.pt")
            best_metrics = {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "sequence_length": args.seq_len,
                "selection_metric": args.selection_metric,
                "selection_score": selection_score,
                "threshold_metric": args.threshold_metric,
                "train_distribution": train_distribution,
                "val_distribution": val_distribution,
                "feature_mean": feature_mean.tolist(),
                "feature_std": feature_std.tolist(),
                **val_metrics,
            }
            marker = "  💾 Best!"
        else:
            marker = ""

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["val_balanced_acc"].append(val_metrics["balanced_acc"])
        history["val_mcc"].append(val_metrics["mcc"])
        history["val_roc_auc"].append(val_metrics["roc_auc"])
        history["val_pr_auc"].append(val_metrics["pr_auc"])
        history["val_threshold"].append(threshold)
        history["val_confusion_matrix"].append(val_metrics["confusion_matrix"])
        history["val_pred_positive_rate"].append(pred_positive_rate)

        print(
            f"Epoch [{epoch:3d}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | "
            f"F1: {val_metrics['f1']:.4f} | Macro-F1: {val_metrics['macro_f1']:.4f} | "
            f"Balanced Acc: {val_metrics['balanced_acc']:.4f} | MCC: {val_metrics['mcc']:.4f} | "
            f"ROC-AUC: {roc_auc_display} | PR-AUC: {pr_auc_display} | "
            f"Thr: {threshold:.3f} | CM: [[{tn}, {fp}], [{fn}, {tp}]] | "
            f"Pred+ Rate: {pred_positive_rate:.3f}{marker}"
        )

        if len(np.unique(val_preds)) == 1:
            predicted_class = int(val_preds[0])
            print(
                f"  Warning: validation predictions collapsed to class {predicted_class}. "
                "This usually indicates imbalance or a weak decision boundary."
            )

    with open("checkpoints/lstm_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open("checkpoints/lstm_metrics.json", "w") as f:
        json.dump(best_metrics, f, indent=2)

    print(f"\n✅ LSTM Training complete! Best validation {args.selection_metric}: {best_score:.4f}")
    print("   Model saved to: checkpoints/lstm_best.pt")
    if best_metrics:
        print("   Metrics saved to: checkpoints/lstm_metrics.json")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ECG LSTM Risk Predictor")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--n-patients", type=int, default=300, help="Synthetic patients")
    parser.add_argument("--seq-len", type=int, default=12, help="Timesteps per sequence")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--real-data", action="store_true", help="Load real PTB-XL data instead of synthetic data")
    parser.add_argument("--ptbxl-dir", type=str, default=None, help="Path to the extracted PTB-XL dataset")
    parser.add_argument("--sampling-rate", type=int, default=100, choices=[100, 500], help="PTB-XL signal sampling rate")
    parser.add_argument("--lead-index", type=int, default=1, help="Lead index to load from each PTB-XL record")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on loaded PTB-XL records")
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="balanced_acc",
        choices=["loss", "acc", "f1", "macro_f1", "balanced_acc", "mcc", "roc_auc", "pr_auc"],
        help="Metric used to choose the best checkpoint",
    )
    parser.add_argument(
        "--threshold-metric",
        type=str,
        default="balanced_acc",
        choices=["f1", "macro_f1", "balanced_acc", "mcc"],
        help="Metric used to tune the validation probability threshold",
    )
    parser.add_argument("--no-class-weights", dest="use_class_weights", action="store_false", help="Disable inverse-frequency class weights in the LSTM loss")
    parser.set_defaults(use_class_weights=True)
    args = parser.parse_args()
    train(args)
