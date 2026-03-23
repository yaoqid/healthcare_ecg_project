"""Data loading and preprocessing for synthetic data and PTB-XL."""

import ast
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import io

try:
    import wfdb
except ImportError:  # pragma: no cover - optional dependency at runtime
    wfdb = None


def ecg_signal_to_image(signal: np.ndarray, image_size: int = 128) -> np.ndarray:
    """Render a 1D ECG trace into a square grayscale image for the CNN."""
    fig, ax = plt.subplots(figsize=(2, 2), dpi=image_size // 2)
    ax.plot(signal, color="black", linewidth=0.5)
    ax.axis("off")
    fig.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert("L")
    img = img.resize((image_size, image_size))
    return np.array(img, dtype=np.float32) / 255.0


class ECGImageDataset(Dataset):
    """Dataset that converts waveform rows into normalized grayscale images."""

    def __init__(self, dataframe: pd.DataFrame, image_size: int = 128, augment: bool = False):
        self.df = dataframe.reset_index(drop=True)
        self.image_size = image_size

        base_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
        if augment:
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

        img_array = ecg_signal_to_image(signal, self.image_size)
        img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
        img_tensor = self.transform(img_pil)

        return img_tensor, torch.tensor(label, dtype=torch.long)


FEATURE_COLUMNS = [
    "heart_rate",
    "pr_interval",
    "qrs_duration",
    "qt_interval",
    "st_elevation",
    "p_wave_amplitude",
    "t_wave_amplitude",
    "rr_variance",
]

PTBXL_TRAIN_FOLDS = tuple(range(1, 9))
PTBXL_VAL_FOLDS = (9,)


def _require_wfdb():
    """Raise a helpful error if wfdb is missing when PTB-XL loading is requested."""
    if wfdb is None:
        raise ImportError(
            "wfdb is required for PTB-XL loading. Install dependencies with "
            "`pip install -r requirements.txt`."
        )


def _parse_scp_codes(value) -> dict:
    """Parse the SCP code dictionary stored as a string in ptbxl_database.csv."""
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return {}
    return ast.literal_eval(value)


def _get_record_path(dataset_path: str, relative_path: str) -> str:
    """Build the wfdb record path without the file extension."""
    return os.path.join(dataset_path, relative_path)


def load_ptbxl_metadata(dataset_path: str, sampling_rate: int = 100) -> pd.DataFrame:
    """Load PTB-XL metadata and map diagnostic superclasses to a binary label."""
    dataset_path = os.path.abspath(dataset_path)
    db_path = os.path.join(dataset_path, "ptbxl_database.csv")
    scp_path = os.path.join(dataset_path, "scp_statements.csv")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"PTB-XL metadata file not found: {db_path}")
    if not os.path.exists(scp_path):
        raise FileNotFoundError(f"PTB-XL statements file not found: {scp_path}")

    ptbxl_df = pd.read_csv(db_path)
    ptbxl_df["scp_codes"] = ptbxl_df["scp_codes"].apply(_parse_scp_codes)

    scp_df = pd.read_csv(scp_path, index_col=0)
    diagnostic_df = scp_df[scp_df["diagnostic"] == 1]
    code_to_superclass = diagnostic_df["diagnostic_class"].dropna().to_dict()

    def diagnostic_superclasses(scp_codes: dict) -> list:
        classes = {code_to_superclass[code] for code in scp_codes if code in code_to_superclass}
        return sorted(classes)

    ptbxl_df["diagnostic_superclasses"] = ptbxl_df["scp_codes"].apply(diagnostic_superclasses)
    ptbxl_df = ptbxl_df[ptbxl_df["diagnostic_superclasses"].map(bool)].copy()
    ptbxl_df["label"] = ptbxl_df["diagnostic_superclasses"].apply(
        lambda classes: 0 if classes == ["NORM"] else 1
    )

    filename_col = "filename_lr" if sampling_rate == 100 else "filename_hr"
    if filename_col not in ptbxl_df.columns:
        raise ValueError(f"Sampling rate {sampling_rate} is unsupported for this PTB-XL metadata.")
    ptbxl_df["record_path"] = ptbxl_df[filename_col].apply(lambda path: _get_record_path(dataset_path, path))

    if "recording_date" in ptbxl_df.columns:
        ptbxl_df["recording_date"] = pd.to_datetime(ptbxl_df["recording_date"], errors="coerce")

    return ptbxl_df


def load_ptbxl_signal(record_path: str, lead_index: int = 1) -> np.ndarray:
    """Load one PTB-XL record and return a single lead as a 1D float array."""
    _require_wfdb()
    signal, _ = wfdb.rdsamp(record_path)
    if signal.ndim == 1:
        return signal.astype(np.float32)
    if lead_index < 0 or lead_index >= signal.shape[1]:
        lead_index = 0
    return signal[:, lead_index].astype(np.float32)


def _deduplicate_peaks(peaks: np.ndarray, min_distance: int) -> np.ndarray:
    """Keep strong peaks while enforcing a refractory period."""
    if len(peaks) == 0:
        return peaks
    filtered = [int(peaks[0])]
    for peak in peaks[1:]:
        if peak - filtered[-1] >= min_distance:
            filtered.append(int(peak))
    return np.asarray(filtered, dtype=np.int32)


def estimate_r_peaks(signal: np.ndarray, sampling_rate: int) -> np.ndarray:
    """Estimate R peaks using a lightweight local-maxima heuristic."""
    centered = signal - np.median(signal)
    scale = np.std(centered) + 1e-6
    normalized = centered / scale
    threshold = max(0.8, np.percentile(normalized, 92))
    candidate_mask = (
        (normalized[1:-1] > normalized[:-2]) &
        (normalized[1:-1] > normalized[2:]) &
        (normalized[1:-1] > threshold)
    )
    candidates = np.where(candidate_mask)[0] + 1
    min_distance = max(1, int(0.25 * sampling_rate))
    return _deduplicate_peaks(candidates, min_distance)


def _window_values(signal: np.ndarray, start: int, end: int) -> np.ndarray:
    start = max(0, start)
    end = min(len(signal), end)
    if start >= end:
        return np.asarray([], dtype=np.float32)
    return signal[start:end]


def extract_ptbxl_features(signal: np.ndarray, sampling_rate: int) -> dict:
    """Estimate simple ECG-inspired features from a single lead."""
    signal = signal.astype(np.float32)
    peaks = estimate_r_peaks(signal, sampling_rate)
    rr_intervals = np.diff(peaks) / float(sampling_rate) if len(peaks) >= 2 else np.asarray([])

    if len(rr_intervals) > 0 and rr_intervals.mean() > 0:
        heart_rate = float(60.0 / rr_intervals.mean())
        rr_variance = float(rr_intervals.var())
    else:
        heart_rate = 70.0
        rr_variance = 0.0

    qrs_durations = []
    st_levels = []
    p_amplitudes = []
    t_amplitudes = []

    for peak in peaks[: min(len(peaks), 20)]:
        baseline_window = _window_values(signal, peak - int(0.25 * sampling_rate), peak - int(0.18 * sampling_rate))
        baseline = float(baseline_window.mean()) if len(baseline_window) else float(np.median(signal))

        qrs_window = _window_values(signal, peak - int(0.08 * sampling_rate), peak + int(0.08 * sampling_rate))
        if len(qrs_window):
            active = np.abs(qrs_window - baseline) > 0.5 * (np.std(signal) + 1e-6)
            qrs_durations.append(active.sum() / sampling_rate * 1000.0)

        st_window = _window_values(signal, peak + int(0.06 * sampling_rate), peak + int(0.12 * sampling_rate))
        if len(st_window):
            st_levels.append(float(st_window.mean() - baseline))

        p_window = _window_values(signal, peak - int(0.20 * sampling_rate), peak - int(0.08 * sampling_rate))
        if len(p_window):
            p_amplitudes.append(float(p_window.max() - baseline))

        t_window = _window_values(signal, peak + int(0.12 * sampling_rate), peak + int(0.32 * sampling_rate))
        if len(t_window):
            t_amplitudes.append(float(t_window.max() - baseline))

    qrs_duration = float(np.mean(qrs_durations)) if qrs_durations else 95.0
    st_elevation = float(np.mean(st_levels)) if st_levels else 0.0
    p_wave_amplitude = float(np.mean(p_amplitudes)) if p_amplitudes else 0.12
    t_wave_amplitude = float(np.mean(t_amplitudes)) if t_amplitudes else 0.25
    pr_interval = float(np.clip(140.0 + rr_variance * 1000.0, 100.0, 240.0))
    qt_interval = float(np.clip(320.0 + qrs_duration + heart_rate * 0.6, 320.0, 520.0))

    return {
        "heart_rate": heart_rate,
        "pr_interval": pr_interval,
        "qrs_duration": qrs_duration,
        "qt_interval": qt_interval,
        "st_elevation": st_elevation,
        "p_wave_amplitude": p_wave_amplitude,
        "t_wave_amplitude": t_wave_amplitude,
        "rr_variance": rr_variance,
    }


def load_ptbxl_cnn_dataframe(
    dataset_path: str,
    sampling_rate: int = 100,
    lead_index: int = 1,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load PTB-XL waveform records into the CNN dataframe format."""
    metadata_df = load_ptbxl_metadata(dataset_path, sampling_rate=sampling_rate)
    if limit is not None:
        metadata_df = metadata_df.head(limit).copy()

    records = []
    for _, row in metadata_df.iterrows():
        signal = load_ptbxl_signal(row["record_path"], lead_index=lead_index)
        records.append({
            "ecg_id": int(row["ecg_id"]),
            "patient_id": int(row["patient_id"]),
            "ecg_signal": signal,
            "label": int(row["label"]),
            "strat_fold": int(row["strat_fold"]),
            "diagnostic_superclasses": row["diagnostic_superclasses"],
        })

    return pd.DataFrame(records)


def load_ptbxl_sequence_dataframe(
    dataset_path: str,
    sampling_rate: int = 100,
    lead_index: int = 1,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Load PTB-XL into the LSTM dataframe format using repeated records per patient.

    Because PTB-XL is not a fixed monthly follow-up dataset, timestamps are derived
    from each patient's recording order.
    """
    metadata_df = load_ptbxl_metadata(dataset_path, sampling_rate=sampling_rate)
    if limit is not None:
        metadata_df = metadata_df.head(limit).copy()

    rows = []
    sorted_df = metadata_df.sort_values(["patient_id", "recording_date", "ecg_id"])
    for patient_id, group in sorted_df.groupby("patient_id"):
        group = group.reset_index(drop=True)
        if len(group) < 2:
            continue

        for timestamp, (_, row) in enumerate(group.iterrows()):
            signal = load_ptbxl_signal(row["record_path"], lead_index=lead_index)
            features = extract_ptbxl_features(signal, sampling_rate=sampling_rate)
            rows.append({
                "patient_id": int(patient_id),
                "timestamp": timestamp,
                "label": int(row["label"]),
                "ecg_id": int(row["ecg_id"]),
                "strat_fold": int(row["strat_fold"]),
                **features,
            })

    return pd.DataFrame(rows)


class ECGSequenceDataset(Dataset):
    """Dataset that turns per-visit feature tables into fixed-length sequences."""

    def __init__(self, dataframe: pd.DataFrame, seq_len: int = 12):
        """Build sliding-window sequences from a patient-level feature table."""
        self.sequences = []
        self.labels    = []
        self.seq_len   = seq_len

        for pid, group in dataframe.groupby("patient_id"):
            group = group.sort_values("timestamp")
            features = group[FEATURE_COLUMNS].values.astype(np.float32)
            label    = int(group["label"].iloc[-1])

            # Overlapping windows increase sample count for patients with longer histories.
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


def generate_synthetic_ecg(n_samples: int = 1000, signal_len: int = 500):
    """Generate simple synthetic waveforms for demo training and UI testing."""
    records = []
    for i in range(n_samples):
        label = np.random.randint(0, 2)
        t = np.linspace(0, 4 * np.pi, signal_len)

        if label == 0:
            signal = (
                0.2 * np.sin(t) +
                1.0 * np.sin(2 * t) +
                0.3 * np.sin(t / 2) +
                0.02 * np.random.randn(signal_len)
            )
        else:
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
        base_hr = 60 + 20 * label + np.random.randn()

        for month in range(seq_len + 6):
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


def get_cnn_dataloaders(df: pd.DataFrame, batch_size: int = 32, val_split: float = 0.2):
    """Create CNN train and validation loaders from a waveform dataframe."""
    if "strat_fold" in df.columns and df["strat_fold"].notna().all():
        train_df = df[df["strat_fold"].isin(PTBXL_TRAIN_FOLDS)].reset_index(drop=True)
        val_df = df[df["strat_fold"].isin(PTBXL_VAL_FOLDS)].reset_index(drop=True)
        if len(train_df) == 0 or len(val_df) == 0:
            raise ValueError("PTB-XL fold-based split produced an empty train/val set.")
    else:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        split = int(len(df) * (1 - val_split))
        train_df, val_df = df[:split], df[split:]

    train_ds = ECGImageDataset(train_df, augment=True)
    val_ds   = ECGImageDataset(val_df,   augment=False)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("CNN dataloaders are empty. Check the dataset path, folds, or limit.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"📊 CNN DataLoaders: {len(train_ds)} train / {len(val_ds)} val samples")
    return train_loader, val_loader


def get_lstm_dataloaders(df: pd.DataFrame, batch_size: int = 32, val_split: float = 0.2, seq_len: int = 12):
    """Create LSTM train and validation loaders from a per-visit feature table."""
    if "strat_fold" in df.columns and df["strat_fold"].notna().all():
        train_df = df[df["strat_fold"].isin(PTBXL_TRAIN_FOLDS)].copy()
        val_df = df[df["strat_fold"].isin(PTBXL_VAL_FOLDS)].copy()
        if len(train_df) == 0 or len(val_df) == 0:
            raise ValueError("PTB-XL fold-based split produced an empty train/val set.")
    else:
        patient_ids = df["patient_id"].unique()
        np.random.shuffle(patient_ids)
        split = int(len(patient_ids) * (1 - val_split))
        train_ids, val_ids = patient_ids[:split], patient_ids[split:]
        train_df = df[df["patient_id"].isin(train_ids)]
        val_df = df[df["patient_id"].isin(val_ids)]

    train_ds = ECGSequenceDataset(train_df, seq_len=seq_len)
    val_ds   = ECGSequenceDataset(val_df,   seq_len=seq_len)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(
            "LSTM dataloaders are empty. On PTB-XL, try reducing --seq-len or increasing the number of loaded records."
        )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    print(f"📊 LSTM DataLoaders: {len(train_ds)} train / {len(val_ds)} val samples")
    return train_loader, val_loader


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
