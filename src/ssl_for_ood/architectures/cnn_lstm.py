from __future__ import annotations

import torch
import torch.nn as nn

from ._base import _SequenceInputMixin


class CNNLSTMRegressor(nn.Module, _SequenceInputMixin):
    def __init__(
        self,
        vocab_size: int,
        cnn_channels: list[int],
        kernel_size: int,
        lstm_hidden_dim: int,
        lstm_layers: int,
        dropout_input: float,
        dropout_hidden: float,
        bidirectional: bool,
        fc_dim: int,
    ) -> None:
        super().__init__()
        if not cnn_channels:
            raise ValueError("cnn_channels must contain at least one value")
        self.vocab_size = vocab_size
        self.input_dropout = nn.Dropout(dropout_input)

        conv_layers: list[nn.Module] = []
        in_channels = vocab_size
        for out_channels in cnn_channels:
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout_hidden),
            ])
            in_channels = out_channels
        self.conv = nn.Sequential(*conv_layers)

        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout_hidden if lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        lstm_output_dim = lstm_hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout_hidden),
            nn.Linear(fc_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_one_hot(x, self.vocab_size)
        x = self.input_dropout(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)
