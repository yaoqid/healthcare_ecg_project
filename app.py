"""
ECG Heart Disease Detection - Streamlit App
--------------------------------------------
The full pipeline in one interactive UI:
  1. Use a synthetic demo patient or load a real PTB-XL record
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
import json
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.cnn_model  import get_model as get_cnn
from models.lstm_model import get_model as get_lstm
from models.llm_assistant import ECGAssistant
from data.data_loader import (
    extract_ptbxl_features,
    ecg_signal_to_image,
    load_ptbxl_metadata,
    load_ptbxl_signal,
    generate_synthetic_ecg,
    generate_synthetic_sequences,
    FEATURE_COLUMNS,
)

st.set_page_config(
    page_title="ECG Heart Disease AI",
    layout="wide",
)

@st.cache_resource
def load_models():
    cnn  = get_cnn(num_classes=2)
    lstm = get_lstm(input_size=8, num_classes=2)
    cnn.eval()
    lstm.eval()
    lstm_metadata = {}

    # Load weights if checkpoint exists
    if os.path.exists("checkpoints/cnn_best.pt"):
        cnn.load_state_dict(torch.load("checkpoints/cnn_best.pt", map_location="cpu"))
        st.sidebar.success("CNN weights loaded")
    else:
        st.sidebar.warning("CNN using random weights - train first!")

    if os.path.exists("checkpoints/lstm_best.pt"):
        lstm.load_state_dict(torch.load("checkpoints/lstm_best.pt", map_location="cpu"))
        st.sidebar.success("LSTM weights loaded")
        if os.path.exists("checkpoints/lstm_metrics.json"):
            try:
                with open("checkpoints/lstm_metrics.json", "r") as f:
                    lstm_metadata = json.load(f)
                st.sidebar.success("LSTM metadata loaded")
            except Exception as exc:
                st.sidebar.warning(f"Could not read LSTM metadata: {exc}")
    else:
        st.sidebar.warning("LSTM using random weights - train first!")

    return cnn, lstm, lstm_metadata


@st.cache_resource
def load_assistant():
    return ECGAssistant()   # reads DEEPSEEK_API_KEY from env


@st.cache_data(show_spinner=False)
def load_ptbxl_metadata_cached(dataset_path: str, sampling_rate: int) -> pd.DataFrame:
    """Cache PTB-XL metadata for interactive browsing in the app."""
    metadata_df = load_ptbxl_metadata(dataset_path, sampling_rate=sampling_rate)
    metadata_df = metadata_df.sort_values(["patient_id", "recording_date", "ecg_id"]).reset_index(drop=True)
    metadata_df["history_order"] = metadata_df.groupby("patient_id").cumcount() + 1
    metadata_df["patient_record_count"] = metadata_df.groupby("patient_id")["ecg_id"].transform("size")
    return metadata_df


@st.cache_data(show_spinner=False)
def load_ptbxl_case(
    dataset_path: str,
    sampling_rate: int,
    lead_index: int,
    ecg_id: int,
    max_history: int,
) -> dict:
    """Load one PTB-XL ECG record plus the selected patient's longitudinal history."""
    metadata_df = load_ptbxl_metadata_cached(dataset_path, sampling_rate=sampling_rate)
    match = metadata_df.loc[metadata_df["ecg_id"] == ecg_id]
    if match.empty:
        raise ValueError(f"ECG ID {ecg_id} was not found in the PTB-XL metadata.")

    selected_row = match.iloc[0]
    signal = load_ptbxl_signal(selected_row["record_path"], lead_index=lead_index)

    patient_records = metadata_df.loc[metadata_df["patient_id"] == selected_row["patient_id"]].copy()
    patient_records = patient_records.sort_values("history_order")
    patient_history = patient_records.loc[
        patient_records["history_order"] <= selected_row["history_order"]
    ].tail(max_history).reset_index(drop=True)
    if patient_history.empty:
        patient_history = selected_row.to_frame().T.reset_index(drop=True)

    feature_rows = []
    for visit_idx, (_, row) in enumerate(patient_history.iterrows(), start=1):
        visit_signal = load_ptbxl_signal(row["record_path"], lead_index=lead_index)
        features = extract_ptbxl_features(visit_signal, sampling_rate=sampling_rate)
        feature_rows.append({
            "Visit": visit_idx,
            "ecg_id": int(row["ecg_id"]),
            "recording_date": None if pd.isna(row["recording_date"]) else str(pd.Timestamp(row["recording_date"]).date()),
            "label": int(row["label"]),
            "diagnostic_superclasses": ", ".join(row["diagnostic_superclasses"]),
            **features,
        })

    sequence_df = pd.DataFrame(feature_rows)
    return {
        "signal": signal.astype(np.float32),
        "sequence": sequence_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32),
        "metadata": {
            "ecg_id": int(selected_row["ecg_id"]),
            "patient_id": int(selected_row["patient_id"]),
            "label": int(selected_row["label"]),
            "diagnostic_superclasses": ", ".join(selected_row["diagnostic_superclasses"]),
            "recording_date": None if pd.isna(selected_row["recording_date"]) else str(pd.Timestamp(selected_row["recording_date"]).date()),
            "history_length": int(len(sequence_df)),
            "available_history": int(selected_row["history_order"]),
            "patient_record_count": int(selected_row["patient_record_count"]),
        },
        "history_table": sequence_df,
    }


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


def predict_lstm(model, sequence: np.ndarray, metadata: dict | None = None) -> dict:
    """Run LSTM on a patient feature sequence and return prediction dict."""
    processed_sequence = np.asarray(sequence, dtype=np.float32)
    threshold = 0.5

    if metadata:
        feature_mean = metadata.get("feature_mean")
        feature_std = metadata.get("feature_std")
        if feature_mean is not None and feature_std is not None:
            mean = np.asarray(feature_mean, dtype=np.float32)
            std = np.asarray(feature_std, dtype=np.float32)
            std = np.where(std < 1e-6, 1.0, std)
            processed_sequence = (processed_sequence - mean) / std
        threshold = float(metadata.get("threshold", 0.5))

    tensor = torch.tensor(processed_sequence, dtype=torch.float32).unsqueeze(0)   # (1, seq, features)

    with torch.no_grad():
        logits       = model(tensor)
        probs        = torch.softmax(logits, dim=1)[0]
        attn_weights = model.get_attention_weights(tensor)[0].numpy()

    risk_score = probs[1].item()   # probability of high risk
    class_id   = int(risk_score >= threshold)

    # Find trend direction
    if len(sequence) < 2:
        trend = "insufficient history"
    else:
        mid = max(1, len(sequence) // 2)
        first_half_hr  = sequence[:mid, 0].mean()   # heart rate col
        second_half_hr = sequence[mid:, 0].mean()
        trend = "increasing" if second_half_hr > first_half_hr else "stable/decreasing"

    return {
        "risk_label":          RISK_NAMES[class_id],
        "risk_score":          risk_score,
        "trend":               trend,
        "attention_weights":   attn_weights.tolist(),
        "most_important_month": int(attn_weights.argmax()) + 1,
        "decision_threshold":  threshold,
    }


def main():
    st.title("ECG Heart Disease Detection & AI Assistant")
    st.markdown(
        "**AI-powered ECG analysis using CNN (image classification) + "
        "LSTM (risk trend prediction) + DeepSeek LLM (plain-English explanations)**"
    )
    st.divider()

    # Load everything
    cnn_model, lstm_model, lstm_metadata = load_models()
    assistant = load_assistant()

    st.sidebar.header("Settings")
    demo_mode = st.sidebar.radio(
        "Data source",
        ["Use demo patient (synthetic)", "Load PTB-XL record"],
    )
    api_key_input = st.sidebar.text_input(
        "DeepSeek API Key (optional)", type="password",
        help="Set DEEPSEEK_API_KEY env var OR paste here"
    )
    if api_key_input:
        assistant.set_api_key(api_key_input)

    st.header("Step 1 - Patient Data")
    patient_note = None
    sequence_axis_label = "Month"
    waveform_title = "Patient ECG Signal"
    trend_title = "Patient Clinical History"
    selected_metadata = None

    if demo_mode == "Use demo patient (synthetic)":
        seed = st.sidebar.slider("Random patient seed", 0, 99, 42)
        np.random.seed(seed)

        # ECG signal
        ecg_df = generate_synthetic_ecg(n_samples=1, signal_len=500)
        signal = ecg_df.iloc[0]["ecg_signal"]
        true_label = ecg_df.iloc[0]["label"]

        # Patient history
        seq_df = generate_synthetic_sequences(n_patients=1, seq_len=18)
        sequence = seq_df[FEATURE_COLUMNS].values[:12]   # 12 timesteps

        patient_note = f"Showing synthetic patient (seed={seed}). True label: **{CLASS_NAMES[true_label]}**"
    else:
        st.markdown("Load a real PTB-XL record from a local extracted dataset folder.")
        default_seq_len = int(lstm_metadata.get("sequence_length", 3)) if lstm_metadata else 3

        # Auto-detect PTB-XL folder: check env var, then default ./ptb-xl
        default_ptbxl = os.getenv("PTBXL_DIR", "")
        if not default_ptbxl:
            candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ptb-xl")
            if os.path.isdir(candidate):
                default_ptbxl = candidate

        ptbxl_dir = default_ptbxl
        if not ptbxl_dir or not os.path.isdir(ptbxl_dir):
            st.error(
                "PTB-XL dataset not found. Expected location: `./ptb-xl` "
                "(relative to the project root). Run `bash scripts/setup_ptbxl.sh` to download it, "
                "or set the `PTBXL_DIR` environment variable to a custom path."
            )
            return

        sampling_rate = st.sidebar.selectbox("PTB-XL sampling rate", [100, 500], index=0)
        lead_index = st.sidebar.number_input("Lead index", min_value=0, max_value=11, value=1, step=1)
        max_history = st.sidebar.slider("History window", 1, 12, max(6, default_seq_len) if max(6, default_seq_len) <= 12 else 12)
        min_history = st.sidebar.slider("Minimum visits required for LSTM", 1, max_history, min(default_seq_len, max_history))

        try:
            metadata_df = load_ptbxl_metadata_cached(ptbxl_dir, sampling_rate)
        except Exception as exc:
            st.error(f"Could not load PTB-XL metadata: {exc}")
            return

        label_filter = st.selectbox("Filter records", ["All records", "Normal only", "Abnormal only"])
        if label_filter == "Normal only":
            filtered_df = metadata_df.loc[metadata_df["label"] == 0].copy()
        elif label_filter == "Abnormal only":
            filtered_df = metadata_df.loc[metadata_df["label"] == 1].copy()
        else:
            filtered_df = metadata_df.copy()

        filtered_df = filtered_df.loc[filtered_df["history_order"] >= min_history].copy()
        if filtered_df.empty:
            st.warning("No PTB-XL records match the current filters and minimum history requirement.")
            return

        filtered_df = filtered_df.sort_values("ecg_id").reset_index(drop=True)
        record_index = st.slider("Browse PTB-XL record", 0, len(filtered_df) - 1, 0)
        selected_row = filtered_df.iloc[record_index]

        try:
            ptbxl_case = load_ptbxl_case(
                ptbxl_dir,
                sampling_rate=int(sampling_rate),
                lead_index=int(lead_index),
                ecg_id=int(selected_row["ecg_id"]),
                max_history=int(max_history),
            )
        except Exception as exc:
            st.error(f"Could not load the selected PTB-XL record: {exc}")
            return

        signal = ptbxl_case["signal"]
        sequence = ptbxl_case["sequence"]
        selected_metadata = ptbxl_case["metadata"]
        sequence_axis_label = "Visit"
        waveform_title = f"PTB-XL ECG Record {selected_metadata['ecg_id']}"
        trend_title = f"PTB-XL Patient History ({selected_metadata['history_length']} visits)"
        patient_note = (
            f"Loaded PTB-XL ECG **{selected_metadata['ecg_id']}** for patient **{selected_metadata['patient_id']}**. "
            f"Dataset label: **{CLASS_NAMES[selected_metadata['label']]}**. "
            f"Diagnostic class: **{selected_metadata['diagnostic_superclasses']}**."
        )

        info_col1, info_col2, info_col3 = st.columns(3)
        info_col1.metric("ECG ID", selected_metadata["ecg_id"])
        info_col2.metric("Patient ID", selected_metadata["patient_id"])
        info_col3.metric("History Length", selected_metadata["history_length"])
        extra_col1, extra_col2 = st.columns(2)
        extra_col1.metric("Available History", selected_metadata["available_history"])
        extra_col2.metric("Total Patient Records", selected_metadata["patient_record_count"])
        if selected_metadata["recording_date"]:
            st.caption(f"Recording date: {selected_metadata['recording_date']}")
        if selected_metadata["history_length"] < default_seq_len:
            st.warning(
                f"This PTB-XL case has only {selected_metadata['history_length']} visit(s) in the selected history window. "
                f"The LSTM was trained with seq_len={default_seq_len}, so temporal interpretation will be limited."
            )
        st.dataframe(
            ptbxl_case["history_table"][["Visit", "ecg_id", "recording_date", "diagnostic_superclasses"] + FEATURE_COLUMNS],
            use_container_width=True,
            hide_index=True,
        )

    if patient_note:
        st.info(patient_note)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ECG Waveform")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(signal, color="#e63946", linewidth=1)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Amplitude (mV)")
        ax.set_title(waveform_title)
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Feature Trends (LSTM Input)")
        feature_df = pd.DataFrame(sequence, columns=FEATURE_COLUMNS)
        feature_df[sequence_axis_label] = range(1, len(feature_df) + 1)
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        if len(feature_df) == 1:
            ax2.scatter(feature_df[sequence_axis_label], feature_df["heart_rate"], label="Heart Rate", color="#457b9d", s=70)
            ax2.scatter(feature_df[sequence_axis_label], feature_df["qt_interval"] / 10, label="QT/10", color="#e9c46a", s=70)
            ax2.set_xlim(0.5, 1.5)
        else:
            ax2.plot(feature_df[sequence_axis_label], feature_df["heart_rate"], label="Heart Rate", color="#457b9d", marker="o")
            ax2.plot(feature_df[sequence_axis_label], feature_df["qt_interval"] / 10, label="QT/10", color="#e9c46a", marker="o")
        ax2.legend()
        ax2.set_xlabel(sequence_axis_label)
        ax2.set_title(trend_title)
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)
        plt.close(fig2)
        if len(feature_df) == 1:
            st.caption("Only one visit is available in the selected PTB-XL history window, so the feature chart shows single-point markers instead of a trend line.")

    st.divider()

    st.header("Step 2 - AI Analysis")

    if st.button("Analyse Patient", type="primary"):
        with st.spinner("Running CNN analysis..."):
            cnn_result = predict_cnn(cnn_model, signal)

        with st.spinner("Running LSTM risk assessment..."):
            lstm_result = predict_lstm(lstm_model, sequence, lstm_metadata)

        # Store in session state so analysis stays visible across reruns (e.g., chat input)
        st.session_state["cnn_result"] = cnn_result
        st.session_state["lstm_result"] = lstm_result
        st.session_state["source_metadata"] = selected_metadata
        assistant.set_patient_context(cnn_result, lstm_result)
        st.session_state["assistant"] = assistant
        st.session_state["chat_history"] = []

    if "cnn_result" in st.session_state and "lstm_result" in st.session_state:
        cnn_result = st.session_state["cnn_result"]
        lstm_result = st.session_state["lstm_result"]

        # Results cards
        col3, col4 = st.columns(2)

        with col3:
            st.metric(
                "CNN Finding",
                cnn_result["prediction"],
                f"Confidence: {cnn_result['confidence']*100:.1f}%"
            )
            st.progress(cnn_result["prob_abnormal"])
            st.caption(f"Abnormal probability: {cnn_result['prob_abnormal']*100:.1f}%")

        with col4:
            time_unit = "Visit" if st.session_state.get("source_metadata") else "Month"
            st.metric(
                "LSTM Risk Level",
                lstm_result["risk_label"],
                f"Score: {lstm_result['risk_score']:.2f} | Trend: {lstm_result['trend']}"
            )
            st.progress(lstm_result["risk_score"])
            st.caption(
                f"Most critical period: {time_unit} {lstm_result['most_important_month']} | "
                f"Threshold: {lstm_result['decision_threshold']:.2f}"
            )

        if st.session_state.get("source_metadata"):
            source_metadata = st.session_state["source_metadata"]
            st.caption(
                f"Reference label from PTB-XL: {CLASS_NAMES[source_metadata['label']]} | "
                f"Diagnostic class: {source_metadata['diagnostic_superclasses']}"
            )

        # Attention weight chart
        axis_label = "Visit" if st.session_state.get("source_metadata") else "Month"
        st.subheader(f"LSTM Attention — Which {axis_label.lower()}s influenced the prediction most?")
        attn = np.array(lstm_result["attention_weights"])
        fig3, ax3 = plt.subplots(figsize=(10, 2))
        x_positions = range(1, len(attn) + 1)
        ax3.bar(x_positions, attn, color=["#e63946" if a == attn.max() else "#457b9d" for a in attn])
        ax3.set_xlabel(axis_label)
        ax3.set_ylabel("Attention Weight")
        ax3.set_title("Temporal Attention Weights (red = most important)")
        st.pyplot(fig3)
        plt.close(fig3)
        if len(attn) == 1:
            st.caption("Attention is 1.0 for the only available visit. To see a meaningful temporal distribution, choose a PTB-XL record with more history.")

    st.divider()

    st.header("Step 3 - Ask the AI Assistant")
    st.markdown(
        "Ask questions in plain English about the ECG results. "
        "Powered by **DeepSeek**."
    )

    if "assistant" not in st.session_state:
        st.info("Click **Analyse Patient** first to enable the assistant.")
    else:
        if st.session_state["assistant"].client is None and "cnn_result" in st.session_state and "lstm_result" in st.session_state:
            st.info("No DeepSeek API key detected. A local summary is shown below, and chat will be disabled until a key is provided.")
            st.markdown(
                ECGAssistant.fallback_summary(
                    st.session_state["cnn_result"],
                    st.session_state["lstm_result"],
                )
            )

        # Display chat history
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        chat_disabled = st.session_state["assistant"].client is None
        if prompt := st.chat_input("Ask a question about the ECG results...", disabled=chat_disabled):
            st.session_state["chat_history"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        reply = st.session_state["assistant"].chat(prompt)
                    except Exception as e:
                        reply = (
                            f"Could not connect to DeepSeek API: {e}\n\n"
                            "Please set your DEEPSEEK_API_KEY environment variable."
                        )
                st.markdown(reply)
                st.session_state["chat_history"].append({"role": "assistant", "content": reply})

    st.divider()
    st.caption(
        "This tool is for educational purposes only. "
        "Always consult a qualified medical professional for clinical decisions."
    )


if __name__ == "__main__":
    main()
