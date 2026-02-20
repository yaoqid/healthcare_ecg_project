"""
Train the CNN Model
--------------------
Run this script to train the ECG image classifier.

Usage:
    python train_cnn.py                        # use synthetic data
    python train_cnn.py --real-data            # use PTB-XL dataset
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

# ── Local imports ─────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.cnn_model import get_model
from data.data_loader import generate_synthetic_ecg, get_cnn_dataloaders


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
    """Evaluate on validation set. Returns loss, accuracy, and all predictions."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), correct / total, all_preds, all_labels




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
    device = get_best_device()
    print(f"🖥️  Training on: {device}")
    os.makedirs("checkpoints", exist_ok=True)

    # ── Data ───────────────────────────────────────────────────────────────
    print("\n📦 Loading data...")
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
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"── Epoch {epoch}/{args.epochs} ──────────────────────────────")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/cnn_best.pt")
            print(f"  💾 New best model saved! Val Acc: {val_acc:.4f}")

        # Log
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}   | Val   Acc: {val_acc:.4f}\n")

    # ── Save training history ───────────────────────────────────────────────
    with open("checkpoints/cnn_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ Training complete! Best Val Accuracy: {best_val_acc:.4f}")
    print("   Model saved to: checkpoints/cnn_best.pt")

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
    args = parser.parse_args()

    train(args)
