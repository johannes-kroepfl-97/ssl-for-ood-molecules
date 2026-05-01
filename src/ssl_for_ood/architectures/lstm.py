'''from __future__ import annotations

import torch
import torch.nn as nn

from ._base import _SequenceInputMixin


class LSTMRegressor(nn.Module, _SequenceInputMixin):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        dropout_input: float,
        dropout_hidden: float,
        bidirectional: bool,
        fc_dim: int,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.input_dropout = nn.Dropout(dropout_input)
        self.lstm = nn.LSTM(
            input_size=vocab_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_hidden if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        output_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Sequential(
            nn.Linear(output_dim, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout_hidden),
            nn.Linear(fc_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_one_hot(x, self.vocab_size)
        x = self.input_dropout(x)
        _, (hn, _) = self.lstm(x)
        if self.lstm.bidirectional:
            out = torch.cat((hn[-2], hn[-1]), dim=1)
        else:
            out = hn[-1]
        return self.fc(out)
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


class LSTMRegressor(nn.Module, _SequenceInputMixin):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        dropout_input: float,
        dropout_hidden: float,
        bidirectional: bool,
        fc_dim: int,
        normalization_strategy: str | None = None,
    ) -> None:
        super().__init__()

        normalization_strategy = validate_normalization_strategy(normalization_strategy)

        self.vocab_size = vocab_size
        self.input_dropout = nn.Dropout(dropout_input)
        self.normalization_strategy = normalization_strategy

        self.input_bn = maybe_batch_norm_last_dim(
            vocab_size,
            normalization_strategy == "input_bn",
        )

        self.lstm = nn.LSTM(
            input_size=vocab_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_hidden if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        output_dim = hidden_dim * (2 if bidirectional else 1)

        self.native_ln = maybe_layer_norm(
            output_dim,
            normalization_strategy == "architecture_native_norm",
        )

        self.final_bn = (
            nn.BatchNorm1d(output_dim)
            if normalization_strategy == "final_bn"
            else nn.Identity()
        )

        self.fc = nn.Sequential(
            nn.Linear(output_dim, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout_hidden),
            nn.Linear(fc_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_one_hot(x, self.vocab_size)
        x = self.input_bn(x)
        x = self.input_dropout(x)

        _, (hn, _) = self.lstm(x)

        if self.lstm.bidirectional:
            out = torch.cat((hn[-2], hn[-1]), dim=1)
        else:
            out = hn[-1]

        out = self.native_ln(out)
        out = self.final_bn(out)
        return self.fc(out)