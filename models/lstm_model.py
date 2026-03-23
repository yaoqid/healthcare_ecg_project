"""LSTM with temporal attention for sequence-level ECG risk prediction."""

import torch
import torch.nn as nn


class ECG_LSTM(nn.Module):
    """Sequence classifier that summarizes an LSTM hidden state with attention."""

    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super(ECG_LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_norm = nn.LayerNorm(input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """Return logits for input sequences shaped `(batch, seq_len, features)`."""
        x = self.input_norm(x)
        lstm_out, _ = self.lstm(x)

        attn_scores = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = (attn_weights * lstm_out).sum(dim=1)

        logits = self.classifier(context)
        return logits

    def get_attention_weights(self, x):
        """Return per-timestep attention weights for model interpretation."""
        x = self.input_norm(x)
        lstm_out, _ = self.lstm(x)
        attn_scores = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        return attn_weights.squeeze(-1)


def get_model(input_size=8, num_classes=2):
    """Instantiate the LSTM model and print its parameter count."""
    model = ECG_LSTM(input_size=input_size, num_classes=num_classes)
    print(f"✅ LSTM Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


if __name__ == "__main__":
    model = get_model()

    dummy_input = torch.randn(4, 12, 8)
    output = model(dummy_input)
    attn   = model.get_attention_weights(dummy_input)

    print(f"Input shape:      {dummy_input.shape}")
    print(f"Output shape:     {output.shape}")
    print(f"Attention shape:  {attn.shape}")
    print("✅ LSTM forward pass OK")
