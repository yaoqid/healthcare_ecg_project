# 🫀 ECG Heart Disease Detection & AI Assistant

An end-to-end ECG project with:
- a CNN for ECG image classification
- an LSTM for temporal risk prediction
- a Streamlit app for interactive analysis
- an LLM assistant for plain-English explanations

This project supports:
- synthetic demo data
- real PTB-XL ECG data

## 📁 Project Structure

```text
healthcare_ecg_project/
├── scripts/
│   ├── setup_ptbxl.sh
│   └── train_ptbxl_models.sh
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
├── requirements.txt
└── .gitignore
```

## 🚀 Setup

Run everything from the project root.

### 1. Create and activate a virtual environment

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install PyTorch

Install the right PyTorch build for your machine from the official selector:
https://pytorch.org/get-started/locally/

Examples:

- **Apple Silicon (MPS)**
  ```bash
  pip install torch torchvision
  ```
- **NVIDIA GPU**
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  ```
- **CPU only**
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```

### 3. Install project dependencies

```bash
pip install -r requirements.txt
```

### 4. Optional: install Kaggle CLI for PTB-XL automation

```bash
pip install kaggle
```

### 5. Sanity check

```bash
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '| mps:', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())"
```

## ⚡ Quick Start

### Option A: Use the project immediately with synthetic data

Train both models:

```bash
python train_cnn.py --epochs 15 --n-samples 1000
python train_lstm.py --epochs 30 --n-patients 300
```

Launch the app:

```bash
streamlit run app.py
```

In the app, choose either:
- `Use demo patient (synthetic)`
- `Enter custom values`

### Option B: Use real PTB-XL data

#### Step 1. Configure Kaggle access

You need one of:
- `~/.kaggle/kaggle.json`
- `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables

You may also need to accept the dataset terms once on Kaggle in your browser.

#### Step 2. Download and prepare PTB-XL automatically

```bash
bash scripts/setup_ptbxl.sh
```

By default this downloads the Kaggle mirror:
- `khyeh0719/ptb-xl-dataset`

and prepares it in:
- `./ptb-xl`

If `./ptb-xl` is already correctly set up, the script skips the download.

#### Step 3. Train both real-data models

```bash
bash scripts/train_ptbxl_models.sh
```

This runs:
- CNN training on PTB-XL
- LSTM training on PTB-XL with the improved balanced-metric pipeline

#### Step 4. Launch the app

```bash
streamlit run app.py
```

In the app:
1. choose `Load PTB-XL record`
2. set the PTB-XL folder to `./ptb-xl` if needed
3. pick a record with enough visit history for LSTM analysis

For more meaningful temporal plots in PTB-XL mode, set:
- `Minimum visits required for LSTM` to `3`

## 🧠 Recommended Commands

### Train CNN on PTB-XL manually

```bash
python train_cnn.py --real-data --ptbxl-dir ./ptb-xl --sampling-rate 100
```

### Train LSTM on PTB-XL manually

```bash
python train_lstm.py --real-data --ptbxl-dir ./ptb-xl --sampling-rate 100 --seq-len 3
```

The improved LSTM pipeline now includes:
- feature standardization
- class-weighted loss
- threshold tuning
- balanced checkpoint selection

### Custom PTB-XL paths

Download to a custom folder:

```bash
bash scripts/setup_ptbxl.sh khyeh0719/ptb-xl-dataset /path/to/ptb-xl
```

Train from a custom folder:

```bash
bash scripts/train_ptbxl_models.sh /path/to/ptb-xl 100 3
```

## 📊 Notes on PTB-XL in This Project

- The CNN uses real PTB-XL waveforms and a binary label: `NORM` vs `non-NORM`.
- The CNN loader uses PTB-XL `strat_fold`: folds `1-8` for training and fold `9` for validation.
- The LSTM does not use official longitudinal follow-up labels because PTB-XL is not a monthly time-series dataset.
- Instead, the LSTM builds short patient histories from repeated PTB-XL records and extracts lightweight waveform-derived features.
- Shorter sequence lengths such as `2` or `3` are usually more practical than long histories on PTB-XL.

## 🛠 Troubleshooting

- **`torch` is missing**
  - Install PyTorch first, then run `pip install -r requirements.txt`.

- **Kaggle download fails**
  - Check that `kaggle` is installed.
  - Check that your Kaggle credentials are configured.
  - Make sure you have accepted the dataset terms on Kaggle.

- **`setup_ptbxl.sh` says the output folder is not empty**
  - Move or remove the existing folder contents, or use a different output path.

- **App PTB-XL mode shows weak temporal plots**
  - Many PTB-XL patients only have one usable visit.
  - Increase `Minimum visits required for LSTM` to `3` in the app sidebar.

- **No GPU available**
  - The project still runs on CPU; training will just be slower.

## ⚠️ Medical Disclaimer

This project is for education and research only. It is not a medical device and must not be used for clinical diagnosis.
