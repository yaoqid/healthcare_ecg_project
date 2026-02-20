"""
🩺 ECG Heart Disease Detection - Streamlit App
-------------------------------------------------
The full pipeline in one interactive UI:
  1. Upload an ECG signal or use a demo
  2. CNN classifies the ECG image
  3. LSTM predicts patient risk trend
  4. DeepSeek LLM explains everything in plain English

Run: streamlit run app.py
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.cnn_model  import get_model as get_cnn
from models.lstm_model import get_model as get_lstm
from models.llm_assistant import ECGAssistant
from data.data_loader import (
    ecg_signal_to_image,
    generate_synthetic_ecg,
    generate_synthetic_sequences,
    FEATURE_COLUMNS,
)


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Page config
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
st.set_page_config(
    page_title="ECG Heart Disease AI",
    page_icon="🫀",
    layout="wide",
)


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Load models (cached so they only load once)
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
@st.cache_resource
def load_models():
    cnn  = get_cnn(num_classes=2)
    lstm = get_lstm(input_size=8, num_classes=2)
    cnn.eval()
    lstm.eval()

    # Load weights if checkpoint exists
    if os.path.exists("checkpoints/cnn_best.pt"):
        cnn.load_state_dict(torch.load("checkpoints/cnn_best.pt", map_location="cpu"))
        st.sidebar.success("✅ CNN weights loaded")
    else:
        st.sidebar.warning("⚠️ CNN using random weights - train first!")

    if os.path.exists("checkpoints/lstm_best.pt"):
        lstm.load_state_dict(torch.load("checkpoints/lstm_best.pt", map_location="cpu"))
        st.sidebar.success("✅ LSTM weights loaded")
    else:
        st.sidebar.warning("⚠️ LSTM using random weights - train first!")

    return cnn, lstm


@st.cache_resource
def load_assistant():
    return ECGAssistant()   # reads DEEPSEEK_API_KEY from env


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Prediction helpers
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
CLASS_NAMES = ["Normal", "Abnormal"]
RISK_NAMES  = ["Low Risk", "High Risk"]


def predict_cnn(model, signal: np.ndarray) -> dict:
    """Run CNN on an ECG signal and return prediction dict."""
    img = ecg_signal_to_image(signal, image_size=128)
    tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)   # (1, 1, 128, 128)
    tensor = (tensor - 0.5) / 0.5   # normalise

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    class_id   = probs.argmax().item()
    confidence = probs[class_id].item()
    return {
        "prediction": CLASS_NAMES[class_id],
        "confidence": confidence,
        "class_id":   class_id,
        "prob_normal":   probs[0].item(),
        "prob_abnormal": probs[1].item(),
    }


def predict_lstm(model, sequence: np.ndarray) -> dict:
    """Run LSTM on a patient feature sequence and return prediction dict."""
    tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)   # (1, seq, features)

    with torch.no_grad():
        logits       = model(tensor)
        probs        = torch.softmax(logits, dim=1)[0]
        attn_weights = model.get_attention_weights(tensor)[0].numpy()

    class_id   = probs.argmax().item()
    risk_score = probs[1].item()   # probability of high risk

    # Find trend direction
    mid = len(sequence) // 2
    first_half_hr  = sequence[:mid, 0].mean()   # heart rate col
    second_half_hr = sequence[mid:, 0].mean()
    trend = "increasing" if second_half_hr > first_half_hr else "stable/decreasing"

    return {
        "risk_label":          RISK_NAMES[class_id],
        "risk_score":          risk_score,
        "trend":               trend,
        "attention_weights":   attn_weights.tolist(),
        "most_important_month": int(attn_weights.argmax()) + 1,
    }


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# UI
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def main():
    st.title("🫀 ECG Heart Disease Detection & AI Assistant")
    st.markdown(
        "**AI-powered ECG analysis using CNN (image classification) + "
        "LSTM (risk trend prediction) + DeepSeek LLM (plain-English explanations)**"
    )
    st.divider()

    # Load everything
    cnn_model, lstm_model = load_models()
    assistant = load_assistant()

    # 鈹€鈹€ Sidebar: controls 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    st.sidebar.header("⚙️ Settings")
    demo_mode = st.sidebar.radio(
        "Data source",
        ["🎲 Use demo patient (synthetic)", "📂 Enter custom values"],
    )
    api_key_input = st.sidebar.text_input(
        "DeepSeek API Key (optional)", type="password",
        help="Set DEEPSEEK_API_KEY env var OR paste here"
    )
    if api_key_input:
        assistant.set_api_key(api_key_input)

    # 鈹€鈹€ Generate or receive data 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    st.header("📊 Step 1 - Patient Data")

    if demo_mode.startswith("🎲"):
        seed = st.sidebar.slider("Random patient seed", 0, 99, 42)
        np.random.seed(seed)

        # ECG signal
        ecg_df = generate_synthetic_ecg(n_samples=1, signal_len=500)
        signal = ecg_df.iloc[0]["ecg_signal"]
        true_label = ecg_df.iloc[0]["label"]

        # Patient history
        seq_df = generate_synthetic_sequences(n_patients=1, seq_len=18)
        sequence = seq_df[FEATURE_COLUMNS].values[:12]   # 12 timesteps

        st.info(f"🎲 Showing synthetic patient (seed={seed}). True label: **{CLASS_NAMES[true_label]}**")
    else:
        st.warning("Custom input coming soon! Using synthetic data for now.")
        ecg_df = generate_synthetic_ecg(n_samples=1, signal_len=500)
        signal = ecg_df.iloc[0]["ecg_signal"]
        seq_df = generate_synthetic_sequences(n_patients=1, seq_len=18)
        sequence = seq_df[FEATURE_COLUMNS].values[:12]

    # 鈹€鈹€ Show ECG plot 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ECG Waveform")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(signal, color="#e63946", linewidth=1)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Amplitude (mV)")
        ax.set_title("Patient ECG Signal")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Feature Trends (LSTM Input)")
        feature_df = pd.DataFrame(sequence, columns=FEATURE_COLUMNS)
        feature_df["Month"] = range(1, 13)
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.plot(feature_df["Month"], feature_df["heart_rate"],   label="Heart Rate", color="#457b9d")
        ax2.plot(feature_df["Month"], feature_df["qt_interval"] / 10, label="QT/10", color="#e9c46a")
        ax2.legend()
        ax2.set_xlabel("Month")
        ax2.set_title("Patient Clinical History")
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)
        plt.close(fig2)

    st.divider()

    # 鈹€鈹€ Run models 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    st.header("🧠 Step 2 - AI Analysis")

    if st.button("🔍 Analyse Patient", type="primary"):
        with st.spinner("Running CNN analysis..."):
            cnn_result = predict_cnn(cnn_model, signal)

        with st.spinner("Running LSTM risk assessment..."):
            lstm_result = predict_lstm(lstm_model, sequence)

        # Store in session state so analysis stays visible across reruns (e.g., chat input)
        st.session_state["cnn_result"] = cnn_result
        st.session_state["lstm_result"] = lstm_result
        assistant.set_patient_context(cnn_result, lstm_result)
        st.session_state["assistant"] = assistant
        st.session_state["chat_history"] = []

    if "cnn_result" in st.session_state and "lstm_result" in st.session_state:
        cnn_result = st.session_state["cnn_result"]
        lstm_result = st.session_state["lstm_result"]

        # Results cards
        col3, col4 = st.columns(2)

        with col3:
            color = "🔶" if cnn_result["class_id"] == 1 else "🟝"
            st.metric(
                f"{color} CNN Finding",
                cnn_result["prediction"],
                f"Confidence: {cnn_result['confidence']*100:.1f}%"
            )
            st.progress(cnn_result["prob_abnormal"])
            st.caption(f"Abnormal probability: {cnn_result['prob_abnormal']*100:.1f}%")

        with col4:
            risk_color = "🔶" if cnn_result["class_id"] == 1 else "🟝"
            st.metric(
                f"{risk_color} LSTM Risk Level",
                lstm_result["risk_label"],
                f"Score: {lstm_result['risk_score']:.2f} | Trend: {lstm_result['trend']}"
            )
            st.progress(lstm_result["risk_score"])
            st.caption(f"Most critical period: Month {lstm_result['most_important_month']}")

        # Attention weight chart
        st.subheader("🔎 LSTM Attention — Which months influenced the prediction most?")
        attn = np.array(lstm_result["attention_weights"])
        fig3, ax3 = plt.subplots(figsize=(10, 2))
        ax3.bar(range(1, 13), attn, color=["#e63946" if a == attn.max() else "#457b9d" for a in attn])
        ax3.set_xlabel("Month")
        ax3.set_ylabel("Attention Weight")
        ax3.set_title("Temporal Attention Weights (red = most important)")
        st.pyplot(fig3)
        plt.close(fig3)

    st.divider()

    # 鈹€鈹€ LLM Chat 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    st.header("💬 Step 3 - Ask the AI Assistant")
    st.markdown(
        "Ask questions in plain English about the ECG results. "
        "Powered by **DeepSeek**."
    )

    if "assistant" not in st.session_state:
        st.info("👆 Click **Analyse Patient** first to enable the assistant.")
    else:
        # Display chat history
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about the ECG results..."):
            st.session_state["chat_history"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        reply = st.session_state["assistant"].chat(prompt)
                    except Exception as e:
                        reply = (
                            f"⚠️ Could not connect to DeepSeek API: {e}\n\n"
                            "Please set your DEEPSEEK_API_KEY environment variable."
                        )
                st.markdown(reply)
                st.session_state["chat_history"].append({"role": "assistant", "content": reply})

    st.divider()
    st.caption(
        "⚠️ This tool is for educational purposes only. "
        "Always consult a qualified medical professional for clinical decisions."
    )


if __name__ == "__main__":
    main()
