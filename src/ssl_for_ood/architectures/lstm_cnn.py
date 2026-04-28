from __future__ import annotations

import torch
import torch.nn as nn

from ._base import _SequenceInputMixin


class LSTMCNNRegressor(nn.Module, _SequenceInputMixin):
    def __init__(
        self,
        vocab_size: int,
        lstm_hidden_dim: int,
        lstm_layers: int,
        dropout_input: float,
        dropout_hidden: float,
        bidirectional: bool,
        cnn_channels: list[int],
        kernel_size: int,
        fc_dim: int,
    ) -> None:
        super().__init__()
        if len(cnn_channels) < 1:
            raise ValueError("cnn_channels must contain at least one value")
        self.vocab_size = vocab_size
        self.input_dropout = nn.Dropout(dropout_input)
        self.lstm = nn.LSTM(
            input_size=vocab_size,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout_hidden if lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        lstm_out_channels = lstm_hidden_dim * (2 if bidirectional else 1)

        conv_layers: list[nn.Module] = []
        in_channels = lstm_out_channels
        for idx, out_channels in enumerate(cnn_channels):
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout_hidden),
            ])
            in_channels = out_channels
        conv_layers.append(nn.AdaptiveAvgPool1d(1))
        self.conv = nn.Sequential(*conv_layers)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cnn_channels[-1], fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout_hidden),
            nn.Linear(fc_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_one_hot(x, self.vocab_size)
        x = self.input_dropout(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.transpose(1, 2)
        conv_out = self.conv(lstm_out)
        return self.fc(conv_out)
