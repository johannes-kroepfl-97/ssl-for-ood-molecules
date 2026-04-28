from __future__ import annotations

import torch
import torch.nn as nn

from ._base import _SequenceInputMixin


class MLPRegressor(nn.Module, _SequenceInputMixin):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        hidden_dims: int | list[int],
        num_layers: int,
        dropout_input: float,
        dropout_hidden: float,
    ) -> None:
        super().__init__()

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims] * num_layers

        if len(hidden_dims) != num_layers:
            raise ValueError(
                f"hidden_dims length must match num_layers. "
                f"Got len(hidden_dims)={len(hidden_dims)}, num_layers={num_layers}."
            )

        self.vocab_size = int(vocab_size)
        self.seq_len = int(seq_len)
        self.input_dim = self.vocab_size * self.seq_len

        layers: list[nn.Module] = [
            nn.Dropout(float(dropout_input)),
        ]

        dims = [self.input_dim] + list(hidden_dims)

        for i in range(num_layers):
            layers.extend(
                [
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(float(dropout_hidden)),
                ]
            )

        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_one_hot(x, self.vocab_size)

        if x.shape[1] != self.seq_len:
            raise ValueError(
                f"Expected sequence length {self.seq_len}, got {x.shape[1]}."
            )

        x = x.reshape(x.shape[0], -1)
        x = self.feature_extractor(x)
        return self.output_layer(x)
