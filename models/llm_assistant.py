"""
LLM Assistant - Powered by DeepSeek API
---------------------------------------
Lets doctors or patients ask plain-English questions about ECG results.
"""

import json
import os
from typing import Optional

from openai import OpenAI


SYSTEM_PROMPT = """You are a helpful, friendly, and empathetic AI medical assistant
specialising in ECG (electrocardiogram) analysis and heart health.

Your role:
1. Explain ECG results in simple, non-technical language.
2. Interpret risk scores and trends for patients and clinicians.
3. Provide general heart health education and guidance.
4. Always remind users that AI results should be confirmed by a qualified doctor.
5. Be reassuring but honest. Never downplay serious symptoms.

When given structured result data (JSON), always reference specific numbers in your explanation.
Format your responses with clear sections using markdown.

IMPORTANT: Always end responses with a reminder that this is AI assistance only
and should not replace professional medical advice.
"""


class ECGAssistant:
    """DeepSeek-powered assistant that explains ECG results in plain English."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
    ):
        """
        Initialise the assistant.
        api_key: your DeepSeek API key (or set DEEPSEEK_API_KEY env variable)
        """
        self._api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self._base_url = base_url
        self.client = OpenAI(api_key=self._api_key, base_url=base_url) if self._api_key else None
        self.conversation_history = []
        self.model = model

    def set_api_key(self, api_key: str):
        """Update API key at runtime from UI input."""
        self._api_key = api_key
        self.client = OpenAI(api_key=self._api_key, base_url=self._base_url)

    def _require_client(self):
        """Ensure API client exists before making requests."""
        if self.client is None:
            raise ValueError(
                "DeepSeek API key not set. Configure DEEPSEEK_API_KEY "
                "or paste the key in the app sidebar."
            )
        return self.client

    def set_patient_context(self, cnn_result: dict, lstm_result: dict):
        """Inject model results as context so the assistant can reference them."""
        context_message = f"""
[PATIENT ECG ANALYSIS RESULTS]

CNN Image Analysis:
  - Finding:     {cnn_result.get('prediction', 'Unknown')}
  - Confidence:  {cnn_result.get('confidence', 0)*100:.1f}%

LSTM Temporal Risk Assessment:
  - Risk Level:          {lstm_result.get('risk_label', 'Unknown')}
  - Risk Score:          {lstm_result.get('risk_score', 0):.2f} / 1.0
  - Trend:               {lstm_result.get('trend', 'Unknown')}
  - Critical Time Point: Month {lstm_result.get('most_important_month', 'N/A')}
    (This is the point in the patient history that most influenced the prediction)

Please be ready to answer questions about these results.
"""
        cnn_label = cnn_result.get("prediction", "Unknown")
        cnn_confidence = cnn_result.get("confidence", 0) * 100
        risk_label = lstm_result.get("risk_label", "Unknown")
        risk_score = lstm_result.get("risk_score", 0)
        trend = lstm_result.get("trend", "Unknown")
        important_month = lstm_result.get("most_important_month", "N/A")

        opening_summary = (
            "Thank you - I've reviewed the patient's ECG analysis results. "
            f"The CNN classified the ECG as {cnn_label.lower()} with {cnn_confidence:.1f}% confidence. "
            f"The LSTM risk model assessed this patient as {risk_label.lower()} "
            f"with a risk score of {risk_score:.2f} and a {trend} trend. "
            f"The most influential point in the history was month {important_month}. "
            "I'm ready to explain what this means."
        )
        self.conversation_history = [
            {"role": "user", "content": context_message},
            {
                "role": "assistant",
                "content": opening_summary,
            },
        ]

    @staticmethod
    def fallback_summary(cnn_result: dict, lstm_result: dict) -> str:
        """Return a local summary when no API key is configured."""
        cnn_label = cnn_result.get("prediction", "Unknown")
        cnn_confidence = cnn_result.get("confidence", 0) * 100
        risk_label = lstm_result.get("risk_label", "Unknown")
        risk_score = lstm_result.get("risk_score", 0)
        trend = lstm_result.get("trend", "Unknown")
        important_month = lstm_result.get("most_important_month", "N/A")

        return (
            "### Quick Summary\n"
            f"- CNN finding: **{cnn_label}** ({cnn_confidence:.1f}% confidence)\n"
            f"- LSTM risk level: **{risk_label}** with score **{risk_score:.2f}**\n"
            f"- Trend over time: **{trend}**\n"
            f"- Most influential time point: **Month {important_month}**\n\n"
            "No DeepSeek API key is configured, so this is a local summary rather than a live LLM explanation.\n\n"
            "This is AI assistance only and should not replace professional medical advice."
        )

    def chat(self, user_message: str) -> str:
        """Send a message and get a response with full conversation context."""
        self.conversation_history.append({"role": "user", "content": user_message})

        client = self._require_client()
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *self.conversation_history,
            ],
        )

        reply = response.choices[0].message.content or ""
        self.conversation_history.append({"role": "assistant", "content": reply})
        return reply

    def reset_conversation(self):
        """Start a fresh conversation (clears history)."""
        self.conversation_history = []
        print("Conversation reset.")

    def quick_summary(self, cnn_result: dict, lstm_result: dict) -> str:
        """Generate a one-shot plain-English summary without interactive chat."""
        prompt = f"""
Please provide a brief, patient-friendly summary (3-4 sentences) of these ECG results:

CNN Result: {json.dumps(cnn_result)}
LSTM Result: {json.dumps(lstm_result)}

Keep it simple - the patient is not medically trained.
"""
        client = self._require_client()
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=300,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""


if __name__ == "__main__":
    cnn_result = {"prediction": "Abnormal", "confidence": 0.87, "class_id": 1}
    lstm_result = {
        "risk_label": "High Risk",
        "risk_score": 0.74,
        "trend": "increasing",
        "most_important_month": 10,
    }

    assistant = ECGAssistant()  # reads DEEPSEEK_API_KEY from environment if configured
    assistant.set_patient_context(cnn_result, lstm_result)

    print("ECG Assistant ready. Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue
        response = assistant.chat(user_input)
        print(f"\nAssistant: {response}\n")
