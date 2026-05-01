'''from __future__ import annotations

import torch
import torch.nn as nn

from ._base import _SequenceInputMixin


class TransformerRegressor(nn.Module, _SequenceInputMixin):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout_input: float,
        dropout_hidden: float,
        pooling: str = "last",
    ) -> None:
        super().__init__()
        if pooling not in {"last", "mean"}:
            raise ValueError("pooling must be 'last' or 'mean'")
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.pooling = pooling
        self.input_dropout = nn.Dropout(dropout_input)
        self.embedding = nn.Linear(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_hidden,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout_hidden),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_one_hot(x, self.vocab_size)
        if x.shape[1] != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {x.shape[1]}")
        x = self.input_dropout(x)
        x = self.embedding(x) + self.positional_encoding
        x_encoded = self.encoder(x)
        if self.pooling == "mean":
            pooled = x_encoded.mean(dim=1)
        else:
            pooled = x_encoded[:, -1, :]
        return self.regressor(pooled)
'''

from __future__ import annotations

import torch
import torch.nn as nn

from ._base import _SequenceInputMixin
from .normalization import (
    maybe_batch_norm_last_dim,
    validate_normalization_strategy,
)


class TransformerRegressor(nn.Module, _SequenceInputMixin):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout_input: float,
        dropout_hidden: float,
        pooling: str = "last",
        normalization_strategy: str | None = None,
    ) -> None:
        super().__init__()

        normalization_strategy = validate_normalization_strategy(normalization_strategy)

        if pooling not in {"last", "mean"}:
            raise ValueError("pooling must be 'last' or 'mean'")

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.pooling = pooling
        self.normalization_strategy = normalization_strategy

        self.input_dropout = nn.Dropout(dropout_input)
        self.embedding = nn.Linear(vocab_size, d_model)

        self.input_bn = maybe_batch_norm_last_dim(
            d_model,
            normalization_strategy == "input_bn",
        )

        self.positional_encoding = nn.Parameter(torch.randn(seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_hidden,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        hidden_dim = max(1, d_model // 2)

        self.final_bn = (
            nn.BatchNorm1d(d_model)
            if normalization_strategy == "final_bn"
            else nn.Identity()
        )

        self.regressor = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_hidden),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_one_hot(x, self.vocab_size)

        if x.shape[1] != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {x.shape[1]}")

        x = self.input_dropout(x)
        x = self.embedding(x)
        x = self.input_bn(x)
        x = x + self.positional_encoding

        x_encoded = self.encoder(x)

        if self.pooling == "mean":
            pooled = x_encoded.mean(dim=1)
        else:
            pooled = x_encoded[:, -1, :]

        pooled = self.final_bn(pooled)
        return self.regressor(pooled)