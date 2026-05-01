'''from __future__ import annotations

import torch
import torch.nn as nn

from ._base import _SequenceInputMixin


class CNNRegressor(nn.Module, _SequenceInputMixin):
    def __init__(
        self,
        vocab_size: int,
        channels: list[int],
        kernel_size: int,
        dropout_input: float,
        dropout_hidden: float,
        fc_dim: int,
    ) -> None:
        super().__init__()
        if vocab_size < 2:
            raise ValueError(f"vocab_size must be >= 2, got {vocab_size}")
        if not channels:
            raise ValueError("channels must contain at least one value")
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")

        self.vocab_size = vocab_size
        self.input_dropout = nn.Dropout(dropout_input)

        conv_layers: list[nn.Module] = []
        in_channels = vocab_size
        for out_channels in channels:
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout_hidden),
            ])
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout_hidden),
            nn.Linear(fc_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_one_hot(x, self.vocab_size)
        x = self.input_dropout(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.pool(x)
        return self.head(x)
'''

from __future__ import annotations

import torch
import torch.nn as nn

from ._base import _SequenceInputMixin
from .normalization import validate_normalization_strategy


class CNNRegressor(nn.Module, _SequenceInputMixin):
    def __init__(
        self,
        vocab_size: int,
        channels: list[int],
        kernel_size: int,
        dropout_input: float,
        dropout_hidden: float,
        fc_dim: int,
        normalization_strategy: str | None = None,
    ) -> None:
        super().__init__()

        normalization_strategy = validate_normalization_strategy(normalization_strategy)

        if vocab_size < 2:
            raise ValueError(f"vocab_size must be >= 2, got {vocab_size}")
        if not channels:
            raise ValueError("channels must contain at least one value")

        self.vocab_size = vocab_size
        self.input_dropout = nn.Dropout(dropout_input)
        self.normalization_strategy = normalization_strategy

        conv_layers: list[nn.Module] = []
        in_channels = vocab_size

        for i, out_channels in enumerate(channels):
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

            elif normalization_strategy == "input_bn" and i == 0:
                conv_layers.append(nn.BatchNorm1d(out_channels))

            conv_layers.extend(
                [
                    nn.ReLU(),
                    nn.Dropout(dropout_hidden),
                ]
            )

            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.final_bn = (
            nn.BatchNorm1d(channels[-1])
            if normalization_strategy == "final_bn"
            else nn.Identity()
        )

        self.head = nn.Sequential(
            nn.Linear(channels[-1], fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout_hidden),
            nn.Linear(fc_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_one_hot(x, self.vocab_size)
        x = self.input_dropout(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        x = self.final_bn(x)
        return self.head(x)