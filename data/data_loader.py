"""
Data Loading & Preprocessing
------------------------------
Handles the PTB-XL ECG dataset for both:
  1. CNN training  — converts ECG signals to 2D images
  2. LSTM training — creates time-series sequences of patient features

Dataset: PTB-XL (https://physionet.org/content/ptb-xl/1.0.3/)
Download: pip install wfdb
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image
import io


# ─────────────────────────────────────────────────────────────────────────────
# 1. ECG → IMAGE  (for CNN)
# ─────────────────────────────────────────────────────────────────────────────

def ecg_signal_to_image(signal: np.ndarray, image_size: int = 128) -> np.ndarray:
    """
    Convert a 1D ECG signal into a 2D grayscale image.

    We plot the signal using matplotlib and save it to a pixel array.
    The CNN then 'sees' the waveform just like a human looking at a printout.

    signal: 1D numpy array of ECG voltage readings
    returns: (image_size, image_size) grayscale numpy array
    """
    fig, ax = plt.subplots(figsize=(2, 2), dpi=image_size // 2)
    ax.plot(signal, color="black", linewidth=0.5)
    ax.axis("off")
    fig.tight_layout(pad=0)

    # Render to in-memory buffer → PIL Image → numpy
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert("L")          # "L" = grayscale
    img = img.resize((image_size, image_size))  # standardise size
    return np.array(img, dtype=np.float32) / 255.0   # normalise to [0, 1]


class ECGImageDataset(Dataset):
    """
    PyTorch Dataset that serves ECG images and labels for the CNN.

    Expected DataFrame columns:
        'ecg_signal'  : 1D numpy array of the ECG waveform
        'label'       : 0 = Normal, 1 = Abnormal
    """

    def __init__(self, dataframe: pd.DataFrame, image_size: int = 128, augment: bool = False):
        self.df = dataframe.reset_index(drop=True)
        self.image_size = image_size

        # Standard normalisation (ImageNet-style adapted for grayscale)
        base_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),   # [0,1] → [-1,1]
        ]
        if augment:
            # Light augmentations — medical images need care here
            aug_transforms = [
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
            ]
            self.transform = transforms.Compose(aug_transforms + base_transforms)
        else:
            self.transform = transforms.Compose(base_transforms)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        signal = row["ecg_signal"]
        label  = int(row["label"])

        # Convert signal to image
        img_array = ecg_signal_to_image(signal, self.image_size)

        # PIL expects HWC, transforms.ToTensor → CHW
        img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
        img_tensor = self.transform(img_pil)

        return img_tensor, torch.tensor(label, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# 2. TIME-SERIES SEQUENCES  (for LSTM)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    "heart_rate",        # beats per minute
    "pr_interval",       # ms (atrial → ventricular conduction)
    "qrs_duration",      # ms (ventricular depolarisation)
    "qt_interval",       # ms (ventricular depolarisation + repolarisation)
    "st_elevation",      # mV (injury marker)
    "p_wave_amplitude",  # mV
    "t_wave_amplitude",  # mV
    "rr_variance",       # heart rate variability proxy
]


class ECGSequenceDataset(Dataset):
    """
    PyTorch Dataset for LSTM training.
    Each sample is a TIME SEQUENCE of a patient's ECG features.

    Expected DataFrame columns:
        patient_id, timestamp, + all FEATURE_COLUMNS, label
    """

    def __init__(self, dataframe: pd.DataFrame, seq_len: int = 12):
        """
        dataframe: sorted by (patient_id, timestamp)
        seq_len:   number of timesteps per sample (e.g. 12 = 12 monthly visits)
        """
        self.sequences = []
        self.labels    = []
        self.seq_len   = seq_len

        for pid, group in dataframe.groupby("patient_id"):
            group = group.sort_values("timestamp")
            features = group[FEATURE_COLUMNS].values.astype(np.float32)
            label    = int(group["label"].iloc[-1])   # label at end of sequence

            # Sliding window — creates multiple samples per patient
            for start in range(0, len(features) - seq_len + 1, seq_len // 2):
                seq = features[start : start + seq_len]
                if len(seq) == seq_len:
                    self.sequences.append(seq)
                    self.labels.append(label)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx],    dtype=torch.long)
        return x, y


# ─────────────────────────────────────────────────────────────────────────────
# 3. Synthetic Data Generator  (for testing without downloading PTB-XL)
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_ecg(n_samples: int = 1000, signal_len: int = 500):
    """
    Generate fake ECG data so you can run the full pipeline immediately
    without downloading the real dataset.

    Normal ECGs:   clean sine-wave-like pattern
    Abnormal ECGs: added noise + irregular intervals
    """
    records = []
    for i in range(n_samples):
        label = np.random.randint(0, 2)   # 0 = normal, 1 = abnormal
        t = np.linspace(0, 4 * np.pi, signal_len)

        if label == 0:
            # Normal: regular PQRST waves
            signal = (
                0.2 * np.sin(t) +               # P wave
                1.0 * np.sin(2 * t) +           # QRS complex
                0.3 * np.sin(t / 2) +           # T wave
                0.02 * np.random.randn(signal_len)  # minor noise
            )
        else:
            # Abnormal: irregular rhythm + higher noise
            signal = (
                0.1 * np.sin(t * np.random.uniform(0.8, 1.2)) +
                1.5 * np.sin(2.5 * t + np.random.uniform(0, 1)) +
                0.5 * np.random.randn(signal_len)
            )

        records.append({"ecg_signal": signal, "label": label})

    return pd.DataFrame(records)


def generate_synthetic_sequences(n_patients: int = 200, seq_len: int = 12):
    """Generate fake patient longitudinal data for LSTM testing."""
    rows = []
    for pid in range(n_patients):
        label = np.random.randint(0, 2)
        base_hr = 60 + 20 * label + np.random.randn()   # higher HR → higher risk

        for month in range(seq_len + 6):  # extra months for sliding window
            rows.append({
                "patient_id"     : pid,
                "timestamp"      : month,
                "label"          : label,
                "heart_rate"     : base_hr + np.random.randn() * 5,
                "pr_interval"    : 160 + label * 20 + np.random.randn() * 10,
                "qrs_duration"   : 90  + label * 20 + np.random.randn() * 8,
                "qt_interval"    : 400 + label * 40 + np.random.randn() * 15,
                "st_elevation"   : label * 0.1 + np.random.randn() * 0.05,
                "p_wave_amplitude": 0.15 + np.random.randn() * 0.03,
                "t_wave_amplitude": 0.3  + np.random.randn() * 0.05,
                "rr_variance"    : 0.05  + label * 0.05 + np.random.randn() * 0.01,
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 4. DataLoader helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_cnn_dataloaders(df: pd.DataFrame, batch_size: int = 32, val_split: float = 0.2):
    """Split dataframe into train/val and return DataLoaders for CNN."""
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)   # shuffle
    split = int(len(df) * (1 - val_split))
    train_df, val_df = df[:split], df[split:]

    train_ds = ECGImageDataset(train_df, augment=True)
    val_ds   = ECGImageDataset(val_df,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"📊 CNN DataLoaders: {len(train_ds)} train / {len(val_ds)} val samples")
    return train_loader, val_loader


def get_lstm_dataloaders(df: pd.DataFrame, batch_size: int = 32, val_split: float = 0.2, seq_len: int = 12):
    """Split patients into train/val and return DataLoaders for LSTM."""
    patient_ids = df["patient_id"].unique()
    np.random.shuffle(patient_ids)
    split = int(len(patient_ids) * (1 - val_split))
    train_ids, val_ids = patient_ids[:split], patient_ids[split:]

    train_ds = ECGSequenceDataset(df[df["patient_id"].isin(train_ids)], seq_len=seq_len)
    val_ds   = ECGSequenceDataset(df[df["patient_id"].isin(val_ids)],   seq_len=seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    print(f"📊 LSTM DataLoaders: {len(train_ds)} train / {len(val_ds)} val samples")
    return train_loader, val_loader


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing synthetic data generation...")
    ecg_df  = generate_synthetic_ecg(n_samples=100)
    seq_df  = generate_synthetic_sequences(n_patients=50)

    cnn_train, cnn_val = get_cnn_dataloaders(ecg_df,  batch_size=8)
    lstm_train, lstm_val = get_lstm_dataloaders(seq_df, batch_size=8)

    imgs, labels = next(iter(cnn_train))
    seqs, risk   = next(iter(lstm_train))

    print(f"CNN batch  — images: {imgs.shape}, labels: {labels.shape}")
    print(f"LSTM batch — seqs:   {seqs.shape}, labels: {risk.shape}")
    print("✅ Data pipeline OK")
