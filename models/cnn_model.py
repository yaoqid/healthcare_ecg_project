"""
CNN Model for ECG Image Classification
----------------------------------------
Takes ECG signal plots (as images) and classifies them into:
  0 = Normal
  1 = Abnormal (potential heart disease)

Beginner-friendly: Every step is commented!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECG_CNN(nn.Module):
    """
    A Convolutional Neural Network that 'sees' ECG images
    and learns to detect abnormal patterns.

    Architecture:
        Input Image (1, 128, 128)
            ↓
        Conv Block 1  → 32 feature maps
            ↓
        Conv Block 2  → 64 feature maps
            ↓
        Conv Block 3  → 128 feature maps
            ↓
        Flatten → Fully Connected Layers
            ↓
        Output: 2 classes (Normal / Abnormal)
    """

    def __init__(self, num_classes=2):
        super(ECG_CNN, self).__init__()

        # ── Block 1: Basic edge/shape detection ──────────────────────────────
        self.conv_block1 = nn.Sequential(
            # Conv2d(in_channels, out_channels, kernel_size)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # grayscale → 32 filters
            nn.BatchNorm2d(32),                             # stabilises training
            nn.ReLU(),                                      # activation (non-linearity)
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                             # halve spatial size: 128→64
            nn.Dropout2d(0.25),                             # regularisation
        )

        # ── Block 2: Pattern combinations ────────────────────────────────────
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                             # 64→32
            nn.Dropout2d(0.25),
        )

        # ── Block 3: Complex ECG feature detection ───────────────────────────
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                             # 32→16
            nn.Dropout2d(0.25),
        )

        # ── Classifier Head ───────────────────────────────────────────────────
        # After 3 pooling layers: 128 × 128 → 16 × 16
        # 128 channels × 16 × 16 pixels = 32768 values
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),  # final output scores
        )

    def forward(self, x):
        """Forward pass: image → prediction"""
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x


def get_model(num_classes=2):
    """Helper to instantiate and return the model."""
    model = ECG_CNN(num_classes=num_classes)
    print(f"✅ CNN Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = get_model()
    dummy_input = torch.randn(4, 1, 128, 128)   # batch of 4 grayscale 128×128 images
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")   # should be (4, 2)
    print("✅ CNN forward pass OK")
