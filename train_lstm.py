"""
Train the LSTM Model
---------------------
Run this script to train the patient risk trend predictor.

Usage:
    python train_lstm.py
    python train_lstm.py --epochs 30 --seq-len 12
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.lstm_model import get_model
from data.data_loader import generate_synthetic_sequences, get_lstm_dataloaders


def train_one_epoch(model, loader, optimizer, criterion, device):
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
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for seqs, labels in loader:
        seqs, labels = seqs.to(device), labels.to(device)
        outputs = model(seqs)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def get_best_device() -> torch.device:
    """Select the best available accelerator in priority: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(args):
    device = get_best_device()
    print(f"🖥️  Training on: {device}")
    os.makedirs("checkpoints", exist_ok=True)

    # ── Data ───────────────────────────────────────────────────────────────
    print("\n📦 Generating synthetic patient sequences...")
    df = generate_synthetic_sequences(n_patients=args.n_patients, seq_len=args.seq_len + 6)
    train_loader, val_loader = get_lstm_dataloaders(df, batch_size=args.batch_size, seq_len=args.seq_len)

    # ── Model ──────────────────────────────────────────────────────────────
    print("\n🧠 Building LSTM model...")
    # input_size = number of features per timestep (8 clinical features)
    model = get_model(input_size=8, num_classes=2).to(device)

    # ── Loss + Optimiser ───────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training Loop ──────────────────────────────────────────────────────
    print(f"\n🚀 Starting LSTM training for {args.epochs} epochs...\n")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/lstm_best.pt")
            marker = "  💾 Best!"
        else:
            marker = ""

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch:3d}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}{marker}")

    with open("checkpoints/lstm_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ LSTM Training complete! Best Val Accuracy: {best_val_acc:.4f}")
    print("   Model saved to: checkpoints/lstm_best.pt")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ECG LSTM Risk Predictor")
    parser.add_argument("--epochs",     type=int,   default=30,   help="Training epochs")
    parser.add_argument("--lr",         type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int,   default=32,   help="Batch size")
    parser.add_argument("--n-patients", type=int,   default=300,  help="Synthetic patients")
    parser.add_argument("--seq-len",    type=int,   default=12,   help="Timesteps per sequence")
    args = parser.parse_args()
    train(args)
