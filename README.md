# ECG Heart Disease Detection & AI Assistant

This project combines:
- a CNN for ECG image classification
- an LSTM for temporal risk prediction
- a Streamlit app for interactive analysis
- an LLM assistant for plain-English explanations

It supports both synthetic demo data and real PTB-XL ECG data.

## Project Structure

```text
healthcare_ecg_project/
├── scripts/
│   ├── setup_ptbxl.sh
│   └── train_ptbxl_models.sh
├── data/
│   ├── __init__.py
│   ├── data_loader.py
│   └── ptb-xl/              # downloaded via setup script
├── models/
│   ├── __init__.py
│   ├── cnn_model.py
│   ├── lstm_model.py
│   └── llm_assistant.py
├── checkpoints/             # created after training
├── train_cnn.py
├── train_lstm.py
├── app.py
├── requirements.txt
├── .env.example
└── .gitignore
```

## Setup

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

Install the correct PyTorch build for your machine from:
https://pytorch.org/get-started/locally/

Examples:

- **Apple Silicon**
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

### 4. Optional: install Kaggle CLI

This is only needed if you want to use the PTB-XL automation scripts.

```bash
pip install kaggle
```

### 5. Sanity check

```bash
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '| mps:', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())"
```

## Quick Start

### Option A: Synthetic demo workflow

Train the demo models:

```bash
python train_cnn.py --epochs 15 --n-samples 1000
python train_lstm.py --epochs 30 --n-patients 300
```

Launch the app:

```bash
streamlit run app.py
```

In the app, choose **Use demo patient (synthetic)**.

### Option B: PTB-XL real-data workflow

#### Step 1. Configure Kaggle access

Do this once before running the PTB-XL setup script.

1. Create or sign in to your Kaggle account:
   - https://www.kaggle.com/
2. Open **Settings** from your profile menu.
3. Scroll to the **API** section.
4. Click **Generate New Token**.
   - Kaggle shows you a token string to copy.
5. Save that token using one of the methods below.

Recommended:

**macOS / Linux: save the token as `~/.kaggle/access_token`**
```bash
mkdir -p ~/.kaggle
printf '%s' 'PASTE_YOUR_KAGGLE_TOKEN_HERE' > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

**Windows (PowerShell): save the token as `$HOME\.kaggle\access_token`**
```powershell
New-Item -ItemType Directory -Force -Path "$HOME\.kaggle"
Set-Content -Path "$HOME\.kaggle\access_token" -NoNewline -Value "PASTE_YOUR_KAGGLE_TOKEN_HERE"
```

Alternative:

**macOS / Linux**
```bash
export KAGGLE_API_TOKEN='PASTE_YOUR_KAGGLE_TOKEN_HERE'
```

**Windows (PowerShell)**
```powershell
$env:KAGGLE_API_TOKEN = "PASTE_YOUR_KAGGLE_TOKEN_HERE"
```

To persist the environment variable:

**macOS / Linux**
```bash
echo "export KAGGLE_API_TOKEN='PASTE_YOUR_KAGGLE_TOKEN_HERE'" >> ~/.zshrc
source ~/.zshrc
```

**Windows (PowerShell)**
```powershell
setx KAGGLE_API_TOKEN "PASTE_YOUR_KAGGLE_TOKEN_HERE"
```

Legacy compatibility:

If you want the older credentials format, create `kaggle.json` manually.

**macOS / Linux**
```bash
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json <<'EOF'
{"username":"YOUR_KAGGLE_USERNAME","key":"PASTE_YOUR_KAGGLE_TOKEN_HERE"}
EOF
chmod 600 ~/.kaggle/kaggle.json
```

**Windows (PowerShell)**
```powershell
New-Item -ItemType Directory -Force -Path "$HOME\.kaggle"
Set-Content -Path "$HOME\.kaggle\kaggle.json" -Value '{"username":"YOUR_KAGGLE_USERNAME","key":"PASTE_YOUR_KAGGLE_TOKEN_HERE"}'
```

Test your setup:

```bash
kaggle datasets files -d khyeh0719/ptb-xl-dataset
```

If that works, Kaggle is configured correctly.

#### Step 2. Download and prepare PTB-XL

```bash
bash scripts/setup_ptbxl.sh
```

By default this downloads:
- `khyeh0719/ptb-xl-dataset`

and prepares it at:
- `./data/ptb-xl`

If that folder is already correctly set up, the script skips the download.

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
1. choose **Load PTB-XL record**
2. set the PTB-XL folder to `./data/ptb-xl`
3. use **Minimum visits required for LSTM = 3** for more meaningful temporal plots

## Manual Commands

Use these if you want to run each stage yourself instead of using the automation scripts.

### Train CNN on PTB-XL

```bash
python train_cnn.py --real-data --ptbxl-dir ./data/ptb-xl --sampling-rate 100
```

### Train LSTM on PTB-XL

```bash
python train_lstm.py --real-data --ptbxl-dir ./data/ptb-xl --sampling-rate 100 --seq-len 3
```

The current LSTM pipeline includes:
- feature standardization
- class-weighted loss
- threshold tuning
- balanced checkpoint selection

### Custom paths

Download to a custom folder:

```bash
bash scripts/setup_ptbxl.sh khyeh0719/ptb-xl-dataset /path/to/ptb-xl
```

Train from a custom folder:

```bash
bash scripts/train_ptbxl_models.sh /path/to/ptb-xl 100 3
```

## Notes on PTB-XL in This Project

- The CNN uses real PTB-XL waveforms and a binary label: `NORM` vs `non-NORM`.
- The CNN loader uses PTB-XL `strat_fold`: folds `1-8` for training and fold `9` for validation.
- The LSTM does not use official longitudinal follow-up labels because PTB-XL is not a fixed monthly time-series dataset.
- Instead, it builds short patient histories from repeated PTB-XL records and extracts lightweight waveform-derived features.
- Shorter sequence lengths such as `2` or `3` are usually more practical than long histories on PTB-XL.

## Troubleshooting

- **`torch` is missing**
  - Install PyTorch first, then run `pip install -r requirements.txt`.

- **Kaggle download fails**
  - Make sure `kaggle` is installed.
  - Make sure your Kaggle token is configured.
  - If you get a `403` error, open the dataset page in your browser and accept the dataset terms once.

- **`setup_ptbxl.sh` says the output folder is not empty**
  - Move or remove the existing folder contents, or use a different output path.

- **PTB-XL mode in the app shows weak temporal plots**
  - Many PTB-XL patients only have one usable visit.
  - Increase **Minimum visits required for LSTM** to `3`.

- **No GPU available**
  - The project still runs on CPU; training will be slower.

## Medical Disclaimer

This project is for education and research only. It is not a medical device and must not be used for clinical diagnosis.
