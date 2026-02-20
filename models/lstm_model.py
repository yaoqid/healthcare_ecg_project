"""
LSTM Model for Patient Risk Trend Prediction
---------------------------------------------
Takes a SEQUENCE of a patient's past ECG measurements over time
and predicts their FUTURE risk level.

Why LSTM?  Regular neural networks treat each reading independently.
LSTMs have 'memory' — they understand that today's reading is connected
to last week's, last month's, etc.

Input:  sequence of shape (batch, timesteps, features)
        e.g. 12 months of measurements, each with 8 features
Output: risk score (0 = low, 1 = high)
"""

import torch
import torch.nn as nn


class ECG_LSTM(nn.Module):
    """
    LSTM-based risk predictor.

    Architecture:
        Input sequence  (batch, 12 timesteps, 8 features)
              ↓
        LSTM (2 stacked layers, hidden=128)
              ↓
        Attention over time (focus on the most important timestep)
              ↓
        Fully Connected
              ↓
        Risk score (0-1)
    """

    def __init__(
        self,
        input_size: int = 8,        # number of features per timestep
        hidden_size: int = 128,      # LSTM memory size
        num_layers: int = 2,         # stacked LSTM layers
        num_classes: int = 2,        # Normal / High-Risk
        dropout: float = 0.3,
    ):
        super(ECG_LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # ── Feature Normalisation ─────────────────────────────────────────────
        self.input_norm = nn.LayerNorm(input_size)

        # ── LSTM Core ─────────────────────────────────────────────────────────
        # batch_first=True means input shape is (batch, seq, features)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,    # only look at past (causal)
        )

        # ── Temporal Attention ────────────────────────────────────────────────
        # Learns WHICH timesteps matter most for prediction
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),       # scalar score per timestep
        )

        # ── Classifier ───────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_size)
        returns: logits (batch_size, num_classes)
        """
        # Normalise inputs
        x = self.input_norm(x)

        # LSTM: output shape → (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)

        # Attention: compute importance weights across timesteps
        attn_scores = self.attention(lstm_out)          # (batch, seq, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # sum to 1 over time
        context = (attn_weights * lstm_out).sum(dim=1)   # weighted sum → (batch, hidden)

        # Classify the weighted context vector
        logits = self.classifier(context)
        return logits

    def get_attention_weights(self, x):
        """
        Returns attention weights so we can explain WHICH
        time period was most important for the prediction.
        """
        x = self.input_norm(x)
        lstm_out, _ = self.lstm(x)
        attn_scores = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        return attn_weights.squeeze(-1)   # (batch, seq_len)


def get_model(input_size=8, num_classes=2):
    """Helper to instantiate the LSTM model."""
    model = ECG_LSTM(input_size=input_size, num_classes=num_classes)
    print(f"✅ LSTM Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = get_model()

    # Simulate: batch of 4 patients, 12 monthly readings, 8 features each
    dummy_input = torch.randn(4, 12, 8)
    output = model(dummy_input)
    attn   = model.get_attention_weights(dummy_input)

    print(f"Input shape:      {dummy_input.shape}")   # (4, 12, 8)
    print(f"Output shape:     {output.shape}")         # (4, 2)
    print(f"Attention shape:  {attn.shape}")           # (4, 12)
    print("✅ LSTM forward pass OK")
