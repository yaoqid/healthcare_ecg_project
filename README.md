# 🫀 ECG Heart Disease Detection & AI Assistant

An end-to-end machine learning project that detects heart-disease patterns from ECG data using **CNN + LSTM + LLM**.

This README is written to help the project run on:
- Windows/Linux PCs with NVIDIA GPU (e.g., RTX 3070 Ti, RTX 5070 Ti)
- macOS (Apple Silicon or Intel)
- CPU-only machines (no GPU)

---

## ✅ Compatibility Goal

The code uses this device fallback order:
1. CUDA (NVIDIA GPU)
2. MPS (Apple Silicon GPU on macOS)
3. CPU

So the same codebase can run across different hardware, with no code rewrite.

---

## 📁 Project Structure

```text
healthcare_ecg_project/
├── data/
│   └── data_loader.py
├── models/
│   ├── cnn_model.py
│   ├── lstm_model.py
│   └── llm_assistant.py
├── checkpoints/             # created after training
├── train_cnn.py
├── train_lstm.py
├── app.py
├── requirements.txt         # non-PyTorch dependencies
└── .gitignore
```

---

## 🚀 Setup (All Computers)

### 1) Create and activate a virtual environment

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install PyTorch for your machine

Install PyTorch first from the official selector:
👉 https://pytorch.org/get-started/locally/

Examples:

- **NVIDIA GPU (CUDA 12.1):**
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```
- **CPU-only (any OS):**
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```
- **macOS Apple Silicon (MPS):**
  ```bash
  pip install torch torchvision
  ```

### 3) Install project dependencies

```bash
pip install -r requirements.txt
```

### 4) Quick sanity check

```bash
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '| mps:', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())"
```

---

## 🏋️ Train Models

```bash
python train_cnn.py --epochs 15 --n-samples 1000
python train_lstm.py --epochs 30 --n-patients 300
```

---

## 🌐 Run the App

```bash
streamlit run app.py
```

Then open: http://localhost:8501

---

## 📊 Data Options

### Option A: Synthetic data (default)
No download needed. Useful for portability testing across machines.

### Option B: PTB-XL real ECG data
- https://physionet.org/content/ptb-xl/1.0.3/
- Install `wfdb` (already in `requirements.txt`)

---

## 🛠 Troubleshooting

- **`torch.cuda.is_available() == False` on NVIDIA PC**
  - Install NVIDIA driver and CUDA-compatible PyTorch wheel.
  - Reinstall torch using the PyTorch website command.

- **macOS is slow**
  - Ensure you're using an Apple Silicon native Python build when possible.
  - Check MPS availability in the sanity command above.

- **No GPU available**
  - This project still runs on CPU; training just takes longer.

---

## ⚠️ Medical Disclaimer

This project is for **education/research only** and is **not** a diagnostic medical device.
