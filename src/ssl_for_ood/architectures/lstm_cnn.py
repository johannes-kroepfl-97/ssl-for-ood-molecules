'''
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
'''

from __future__ import annotations

import torch
import torch.nn as nn

from ._base import _SequenceInputMixin
from .normalization import (
    maybe_batch_norm_last_dim,
    maybe_layer_norm,
    validate_normalization_strategy,
)


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
        normalization_strategy: str | None = None,
    ) -> None:
        super().__init__()

        normalization_strategy = validate_normalization_strategy(normalization_strategy)

        if len(cnn_channels) < 1:
            raise ValueError("cnn_channels must contain at least one value")

        self.vocab_size = vocab_size
        self.input_dropout = nn.Dropout(dropout_input)
        self.normalization_strategy = normalization_strategy

        self.input_bn = maybe_batch_norm_last_dim(
            vocab_size,
            normalization_strategy == "input_bn",
        )

        self.lstm = nn.LSTM(
            input_size=vocab_size,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout_hidden if lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_out_channels = lstm_hidden_dim * (2 if bidirectional else 1)

        self.native_lstm_ln = maybe_layer_norm(
            lstm_out_channels,
            normalization_strategy == "architecture_native_norm",
        )

        conv_layers: list[nn.Module] = []
        in_channels = lstm_out_channels

        for i, out_channels in enumerate(cnn_channels):
            conv_layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                )
            )

            if normalization_strategy == "architecture_native_norm":
                conv_layers.append(nn.BatchNorm1d(out_channels))

            conv_layers.extend(
                [
                    nn.ReLU(),
                    nn.Dropout(dropout_hidden),
                ]
            )

            in_channels = out_channels

        conv_layers.append(nn.AdaptiveAvgPool1d(1))
        self.conv = nn.Sequential(*conv_layers)

        self.final_bn = (
            nn.BatchNorm1d(cnn_channels[-1])
            if normalization_strategy == "final_bn"
            else nn.Identity()
        )

        self.fc = nn.Sequential(
            nn.Linear(cnn_channels[-1], fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout_hidden),
            nn.Linear(fc_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_one_hot(x, self.vocab_size)
        x = self.input_bn(x)
        x = self.input_dropout(x)

        lstm_out, _ = self.lstm(x)
        lstm_out = self.native_lstm_ln(lstm_out)

        lstm_out = lstm_out.transpose(1, 2)
        conv_out = self.conv(lstm_out).squeeze(-1)

        conv_out = self.final_bn(conv_out)
        return self.fc(conv_out)
