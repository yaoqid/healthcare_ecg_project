"""
LLM Assistant — Powered by Claude API
---------------------------------------
Lets doctors or patients ask plain-English questions about ECG results.

Example questions:
  "What does it mean that my ECG shows an abnormal pattern?"
  "My risk trend has been increasing for 3 months. Should I be worried?"
  "Explain the attention weights — which part of my history matters most?"
"""

import anthropic
import json
from typing import Optional


# ── System prompt: gives Claude its role ─────────────────────────────────────
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

⚠️ IMPORTANT: Always end responses with a reminder that this is AI assistance only
and should not replace professional medical advice.
"""


class ECGAssistant:
    """Claude-powered assistant that explains ECG results in plain English."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialise the assistant.
        api_key: your Anthropic API key (or set ANTHROPIC_API_KEY env variable)
        """
        # Uses ANTHROPIC_API_KEY env var automatically if api_key is None
        self.client = anthropic.Anthropic(api_key=api_key)
        self.conversation_history = []   # keeps track of the whole conversation
        self.model = "claude-opus-4-6"

    def set_patient_context(self, cnn_result: dict, lstm_result: dict):
        """
        Inject the model results as context so Claude can reference them.

        cnn_result example:
            {"prediction": "Abnormal", "confidence": 0.87, "class_id": 1}

        lstm_result example:
            {"risk_label": "High Risk", "risk_score": 0.74,
             "trend": "increasing", "most_important_month": 10}
        """
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
        # Add as the first user message so Claude has the context
        self.conversation_history = [
            {"role": "user", "content": context_message},
            {"role": "assistant", "content": (
                "Thank you — I've reviewed the patient's ECG analysis results. "
                "I can see the CNN detected an abnormal ECG pattern with high confidence, "
                "and the LSTM risk model shows an increasing risk trend. "
                "I'm ready to explain these findings. What would you like to know?"
            )},
        ]

    def chat(self, user_message: str) -> str:
        """
        Send a message and get a response.
        Maintains conversation history for context-aware responses.
        """
        # Add the new user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Call Claude API
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=self.conversation_history,
        )

        # Extract the reply text
        reply = response.content[0].text

        # Add assistant reply to history (for multi-turn conversation)
        self.conversation_history.append({
            "role": "assistant",
            "content": reply
        })

        return reply

    def reset_conversation(self):
        """Start a fresh conversation (clears history)."""
        self.conversation_history = []
        print("🔄 Conversation reset.")

    def quick_summary(self, cnn_result: dict, lstm_result: dict) -> str:
        """
        Generate a one-shot plain-English summary without interactive chat.
        Good for showing a summary card in the UI.
        """
        prompt = f"""
Please provide a brief, patient-friendly summary (3-4 sentences) of these ECG results:

CNN Result: {json.dumps(cnn_result)}
LSTM Result: {json.dumps(lstm_result)}

Keep it simple — the patient is not medically trained.
"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


# ── Demo / manual test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example results from our CNN + LSTM models
    cnn_result  = {"prediction": "Abnormal", "confidence": 0.87, "class_id": 1}
    lstm_result = {"risk_label": "High Risk", "risk_score": 0.74,
                   "trend": "increasing", "most_important_month": 10}

    assistant = ECGAssistant()   # reads ANTHROPIC_API_KEY from environment
    assistant.set_patient_context(cnn_result, lstm_result)

    print("🩺 ECG Assistant ready. Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue
        response = assistant.chat(user_input)
        print(f"\nAssistant: {response}\n")
