"""
Train the CNN Model
--------------------
Run this script to train the ECG image classifier.

Usage:
    python train_cnn.py                        # use synthetic data
    python train_cnn.py --real-data --ptbxl-dir /path/to/ptb-xl
    python train_cnn.py --epochs 20 --lr 0.001
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import json
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

# ── Local imports ─────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.cnn_model import get_model
from data.data_loader import generate_synthetic_ecg, get_cnn_dataloaders, load_ptbxl_cnn_dataframe


# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    """Run one epoch of training. Returns avg loss and accuracy."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping — prevents exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 10 == 0:
            print(f"  Step [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {loss.item():.4f}")

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on validation set. Returns loss, accuracy, labels, predictions, and scores."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        probs = torch.softmax(outputs, dim=1)

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

    return total_loss / len(loader), correct / total, all_labels, all_preds, all_probs


def compute_metrics(labels, preds, probs):
    """Compute a compact metric set for binary ECG classification."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def set_seed(seed: int):
    """Make synthetic runs reproducible."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    # ── Setup ──────────────────────────────────────────────────────────────
    set_seed(args.seed)
    device = get_best_device()
    print(f"🖥️  Training on: {device}")
    os.makedirs("checkpoints", exist_ok=True)

    # ── Data ───────────────────────────────────────────────────────────────
    print("\n📦 Loading data...")
    if args.real_data:
        if not args.ptbxl_dir:
            raise ValueError("--ptbxl-dir is required when using --real-data.")
        df = load_ptbxl_cnn_dataframe(
            args.ptbxl_dir,
            sampling_rate=args.sampling_rate,
            lead_index=args.lead_index,
            limit=args.limit,
        )
        print(
            f"Loaded {len(df)} PTB-XL records at {args.sampling_rate}Hz "
            f"using lead index {args.lead_index}."
        )
    else:
        df = generate_synthetic_ecg(n_samples=args.n_samples)
    train_loader, val_loader = get_cnn_dataloaders(df, batch_size=args.batch_size)

    # ── Model ──────────────────────────────────────────────────────────────
    print("\n🧠 Building CNN model...")
    model = get_model(num_classes=2).to(device)

    # ── Loss, Optimiser, Scheduler ─────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    # ── Training Loop ──────────────────────────────────────────────────────
    print(f"\n🚀 Starting training for {args.epochs} epochs...\n")
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_roc_auc": [],
    }
    best_val_acc = 0.0
    best_metrics = {}

    for epoch in range(1, args.epochs + 1):
        print(f"── Epoch {epoch}/{args.epochs} ──────────────────────────────")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_labels, val_preds, val_probs = evaluate(model, val_loader, criterion, device)
        val_metrics = compute_metrics(val_labels, val_preds, val_probs)
        roc_auc_display = "N/A" if val_metrics["roc_auc"] is None else f"{val_metrics['roc_auc']:.4f}"

        scheduler.step(val_loss)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/cnn_best.pt")
            best_metrics = {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
                **val_metrics,
            }
            print(f"  💾 New best model saved! Val Acc: {val_acc:.4f}")

        # Log
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_roc_auc"].append(val_metrics["roc_auc"])

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(
            f"  Val   Loss: {val_loss:.4f}   | Val   Acc: {val_acc:.4f} | "
            f"Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | "
            f"F1: {val_metrics['f1']:.4f} | ROC-AUC: {roc_auc_display}\n"
        )

    # ── Save training history ───────────────────────────────────────────────
    with open("checkpoints/cnn_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open("checkpoints/cnn_metrics.json", "w") as f:
        json.dump(best_metrics, f, indent=2)

    print(f"\n✅ Training complete! Best Val Accuracy: {best_val_acc:.4f}")
    print("   Model saved to: checkpoints/cnn_best.pt")
    if best_metrics:
        print("   Metrics saved to: checkpoints/cnn_metrics.json")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ECG CNN Classifier")
    parser.add_argument("--epochs",    type=int,   default=15,    help="Number of training epochs")
    parser.add_argument("--lr",        type=float, default=1e-3,  help="Learning rate")
    parser.add_argument("--batch-size",type=int,   default=32,    help="Batch size")
    parser.add_argument("--n-samples", type=int,   default=1000,  help="Synthetic samples to generate")
    parser.add_argument("--seed",      type=int,   default=42,    help="Random seed")
    parser.add_argument("--real-data", action="store_true", help="Load real PTB-XL data instead of synthetic data")
    parser.add_argument("--ptbxl-dir", type=str,   default=None,  help="Path to the extracted PTB-XL dataset")
    parser.add_argument("--sampling-rate", type=int, default=100, choices=[100, 500], help="PTB-XL signal sampling rate")
    parser.add_argument("--lead-index", type=int, default=1, help="Lead index to load from each PTB-XL record")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on loaded PTB-XL records")
    args = parser.parse_args()

    train(args)
