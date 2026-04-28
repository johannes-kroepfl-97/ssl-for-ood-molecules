from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl


class LightningSequenceRegressor(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        training_config: dict[str, Any],
        y_scaler: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.training_config = training_config
        self.loss_name = str(training_config.get("loss", "mse")).lower()
        self.loss_fn = nn.MSELoss() if self.loss_name == "mse" else nn.L1Loss()
        self.y_scaler = y_scaler

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _compute_metrics(self, preds: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.y_scaler is not None:
            mean = torch.tensor(self.y_scaler["mean"], dtype=preds.dtype, device=preds.device)
            std = torch.tensor(self.y_scaler["std"], dtype=preds.dtype, device=preds.device)
            preds_eval = preds * std + mean
            targets_eval = targets * std + mean
        else:
            preds_eval = preds
            targets_eval = targets
        mae = torch.mean(torch.abs(preds_eval - targets_eval))
        return preds_eval, mae

    def _shared_eval_step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        x, y = batch
        preds = self.model(x)
        loss = self.loss_fn(preds, y)
        _, mae = self._compute_metrics(preds, y)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=(stage != "train"), add_dataloader_idx=False)
        self.log(f"{stage}_mae", mae, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_eval_step(batch, stage="train")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        stage = "val_id" if dataloader_idx == 0 else "val_ood"
        return self._shared_eval_step(batch, stage=stage)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_eval_step(batch, stage="test")

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = AdamW(
            self.parameters(),
            lr=float(self.training_config["learning_rate"]),
            weight_decay=float(self.training_config.get("weight_decay", 0.0)),
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(self.training_config.get("lr_scheduler_factor", 0.5)),
            patience=int(self.training_config.get("lr_scheduler_patience", 5)),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_id_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
