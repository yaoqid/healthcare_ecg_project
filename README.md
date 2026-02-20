# 🫀 ECG Heart Disease Detection & AI Assistant

An end-to-end machine learning project that detects heart disease from ECG data using **CNN + LSTM + LLM**.

---

## 🎯 What This Project Does

| Step | Model | Task |
|------|-------|------|
| 1 | **CNN** | Classifies ECG signal images → Normal / Abnormal |
| 2 | **LSTM** | Predicts patient risk trend from 12 months of clinical history |
| 3 | **LLM (Claude)** | Explains results in plain English via interactive chat |

---

## 📁 Project Structure

```
healthcare_ecg_project/
├── data/
│   └── data_loader.py       # Data loading, preprocessing, synthetic data generator
├── models/
│   ├── cnn_model.py         # CNN architecture (image classifier)
│   ├── lstm_model.py        # LSTM architecture (risk predictor with attention)
│   └── llm_assistant.py     # Claude-powered chat assistant
├── checkpoints/             # Saved model weights (created after training)
├── train_cnn.py             # CNN training script
├── train_lstm.py            # LSTM training script
├── app.py                   # Streamlit web app (full pipeline)
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Your Anthropic API Key (for the LLM assistant)

```bash
# Linux/Mac
export ANTHROPIC_API_KEY="your-key-here"

# Windows
set ANTHROPIC_API_KEY=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
```

Get a free API key at: https://console.anthropic.com

### 3. Train the Models

```bash
# Train the CNN (ECG image classifier) — ~5 min on CPU
python train_cnn.py --epochs 15 --n-samples 1000

# Train the LSTM (risk trend predictor) — ~3 min on CPU
python train_lstm.py --epochs 30 --n-patients 300
```

### 4. Run the App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser. 🎉

---

## 🧠 Model Details

### CNN — ECG Image Classifier

```
Input: ECG signal → rendered as 128×128 grayscale image
Architecture: 3 Conv blocks (32→64→128 filters) + 3 FC layers
Output: Normal (0) or Abnormal (1) with confidence score
```

**Key idea:** We convert 1D ECG signals into 2D images so the CNN can
detect visual patterns (like irregular peaks, ST elevation) the same way
a cardiologist reads a printed ECG strip.

### LSTM — Patient Risk Trend Predictor

```
Input: 12 monthly readings × 8 clinical features
       (heart rate, PR interval, QRS duration, QT interval,
        ST elevation, P/T wave amplitudes, RR variance)
Architecture: 2-layer LSTM + Temporal Attention + 2 FC layers
Output: Low Risk (0) or High Risk (1) + attention weights
```

**Key idea:** The LSTM reads the patient's history sequentially,
with its internal "memory cells" tracking how risk evolves over time.
The **attention mechanism** highlights which months were most important,
giving explainability.

### LLM — Claude Assistant

```
Model: claude-opus-4-6
Mode:  Multi-turn conversational chat
Role:  Explains CNN + LSTM results in plain, patient-friendly English
```

**Key idea:** Raw model outputs (class IDs, confidence scores) are
injected as context into Claude, which then fields natural-language
questions from patients or clinicians.

---

## 📊 Dataset

### Option A: Synthetic Data (default, no download needed)
The project includes a synthetic ECG generator for testing the full pipeline
immediately. Synthetic data won't give real clinical accuracy but lets you
verify the code works end-to-end.

### Option B: PTB-XL (real ECG data, recommended)
- **21,799 ECG records** from real patients
- Free, publicly available on PhysioNet
- Download: https://physionet.org/content/ptb-xl/1.0.3/

```bash
# Install the wfdb library first
pip install wfdb

# Then download
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
```

---

## 📈 Expected Results

With synthetic data (sanity check):
- CNN: ~75-85% val accuracy
- LSTM: ~80-90% val accuracy

With real PTB-XL data (after fine-tuning):
- CNN: ~85-92% accuracy
- LSTM: ~80-88% accuracy

---

## 💬 Example Chat Questions

Once you've run the analysis, try asking the assistant:

- *"What does an abnormal ECG pattern mean?"*
- *"Is my risk score of 0.74 dangerous?"*
- *"Why was month 10 the most important in my history?"*
- *"What lifestyle changes can reduce my heart risk?"*
- *"Should I see a doctor urgently?"*

---

## 🔧 Customisation

### Add more classes
Change `num_classes=2` to `num_classes=5` in `get_model()` calls to classify
specific conditions (e.g. Normal / LBBB / RBBB / ST-elevation / Atrial Fibrillation).

### Use real PTB-XL data
Replace `generate_synthetic_ecg()` in `train_cnn.py` with a real PTB-XL loader
using the `wfdb` library. The rest of the pipeline stays the same.

### Improve CNN accuracy
- Add more data augmentation in `ECGImageDataset`
- Try a pretrained ResNet backbone with `torchvision.models.resnet18()`
- Increase image size from 128→224

---

## ⚠️ Disclaimer

This tool is for **educational purposes only**. It should not be used for
clinical diagnosis or medical decision-making. Always consult a qualified
medical professional.
