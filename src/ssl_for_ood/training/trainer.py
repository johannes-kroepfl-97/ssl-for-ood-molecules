from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import MLFlowLogger
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import MLFlowLogger

import yaml

from ssl_for_ood.architectures import (
    CNNLSTMRegressor,
    CNNRegressor,
    LSTMCNNRegressor,
    LSTMRegressor,
    MLPRegressor,
    TransformerRegressor,
)
from ssl_for_ood.training.datasets import SequenceDataModule
from ssl_for_ood.training.lightning_module import LightningSequenceRegressor


@dataclass
class RunArtifacts:
    run_dir: Path
    best_checkpoint_path: Path | None
    model_state_dict_path: Path
    metrics_path: Path
    config_path: Path
    y_scaler_path: Path | None
    mlflow_run_id_path: Path | None


class NullMlflowLogger:
    def __init__(self) -> None:
        self.run_id: str | None = None
    def log_hyperparams(self, params: dict[str, Any]) -> None:
        return None
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} must contain a dictionary at the top level.")
    return data


def load_config(config_path: str | Path, override_path: str | Path | None = None) -> dict[str, Any]:
    config = load_yaml(config_path)
    if override_path is not None:
        override = load_yaml(override_path)
        config = deep_update(config, override)
    return config


def flatten_dict(d: dict[str, Any], parent_key: str = "") -> dict[str, Any]:
    items: dict[str, Any] = {}
    for key, value in d.items():
        new_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key))
        else:
            items[new_key] = value
    return items


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_run_dir(base_output_dir: str | Path, model_name: str, dataset_name: str, run_name: str | None = None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = run_name or f"{timestamp}"
    return ensure_dir(Path(base_output_dir) / model_name / dataset_name / run_id)


def create_mlflow_logger(config: dict[str, Any], model_name: str, dataset_name: str):
    mlflow_cfg = config.get("mlflow", {})
    if not mlflow_cfg.get("enabled", True):
        return NullMlflowLogger()
    tracking_uri = mlflow_cfg.get("tracking_uri")
    experiment_name = mlflow_cfg.get("experiment_name", f"ssl-ood-{model_name}")
    run_name = mlflow_cfg.get("run_name", f"{model_name}__{dataset_name}")
    try:
        return MLFlowLogger(experiment_name=experiment_name, tracking_uri=tracking_uri, run_name=run_name)
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] Could not initialize MLflow logger: {exc}")
        return NullMlflowLogger()


def evaluate_regression_model(
    lightning_module: LightningSequenceRegressor,
    dataloader,
    y_scaler: dict[str, float] | None,
) -> dict[str, float]:
    device = lightning_module.device
    model = lightning_module.model.eval().to(device)
    preds_all = []
    targets_all = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            preds_all.append(preds)
            targets_all.append(y)
    preds_cat = torch.cat(preds_all, dim=0)
    targets_cat = torch.cat(targets_all, dim=0)
    if y_scaler is not None:
        mean = torch.tensor(y_scaler["mean"], dtype=preds_cat.dtype, device=preds_cat.device)
        std = torch.tensor(y_scaler["std"], dtype=preds_cat.dtype, device=preds_cat.device)
        preds_eval = preds_cat * std + mean
        targets_eval = targets_cat * std + mean
    else:
        preds_eval = preds_cat
        targets_eval = targets_cat
    abs_err = torch.abs(preds_eval - targets_eval)
    mae = torch.mean(abs_err).item()
    std_abs_err = torch.std(abs_err, unbiased=False).item()
    mse = torch.mean((preds_eval - targets_eval) ** 2).item()
    return {"mae": mae, "std_abs_error": std_abs_err, "mse": mse, "n_samples": int(targets_eval.shape[0])}


def save_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def save_yaml(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return path


def build_model(model_name: str, model_cfg: dict[str, Any], vocab_size: int, seq_len: int):
    if model_name == "mlp":
        return MLPRegressor(
            vocab_size=vocab_size,
            seq_len=seq_len,
            hidden_dims=model_cfg["hidden_dims"],
            num_layers=int(model_cfg["num_layers"]),
            dropout_input=float(model_cfg.get("dropout_input", 0.0)),
            dropout_hidden=float(model_cfg.get("dropout_hidden", 0.0)),
        )
    if model_name == "cnn":
        return CNNRegressor(
            vocab_size=vocab_size,
            channels=list(model_cfg["channels"]),
            kernel_size=int(model_cfg["kernel_size"]),
            dropout_input=float(model_cfg.get("dropout_input", 0.0)),
            dropout_hidden=float(model_cfg.get("dropout_hidden", 0.0)),
            fc_dim=int(model_cfg["fc_dim"]),
        )
    if model_name == "lstm":
        return LSTMRegressor(
            vocab_size=vocab_size,
            hidden_dim=int(model_cfg["hidden_dim"]),
            num_layers=int(model_cfg["num_layers"]),
            dropout_input=float(model_cfg.get("dropout_input", 0.0)),
            dropout_hidden=float(model_cfg.get("dropout_hidden", 0.0)),
            bidirectional=bool(model_cfg.get("bidirectional", False)),
            fc_dim=int(model_cfg["fc_dim"]),
        )
    if model_name == "cnn_lstm":
        return CNNLSTMRegressor(
            vocab_size=vocab_size,
            cnn_channels=list(model_cfg["cnn_channels"]),
            kernel_size=int(model_cfg["kernel_size"]),
            lstm_hidden_dim=int(model_cfg["lstm_hidden_dim"]),
            lstm_layers=int(model_cfg["lstm_layers"]),
            dropout_input=float(model_cfg.get("dropout_input", 0.0)),
            dropout_hidden=float(model_cfg.get("dropout_hidden", 0.0)),
            bidirectional=bool(model_cfg.get("bidirectional", False)),
            fc_dim=int(model_cfg["fc_dim"]),
        )
    if model_name == "lstm_cnn":
        return LSTMCNNRegressor(
            vocab_size=vocab_size,
            lstm_hidden_dim=int(model_cfg["lstm_hidden_dim"]),
            lstm_layers=int(model_cfg["lstm_layers"]),
            dropout_input=float(model_cfg.get("dropout_input", 0.0)),
            dropout_hidden=float(model_cfg.get("dropout_hidden", 0.0)),
            bidirectional=bool(model_cfg.get("bidirectional", False)),
            cnn_channels=list(model_cfg["cnn_channels"]),
            kernel_size=int(model_cfg["kernel_size"]),
            fc_dim=int(model_cfg["fc_dim"]),
        )
    if model_name == "transformer":
        return TransformerRegressor(
            vocab_size=vocab_size,
            seq_len=seq_len,
            d_model=int(model_cfg["d_model"]),
            nhead=int(model_cfg["nhead"]),
            num_layers=int(model_cfg["num_layers"]),
            dim_feedforward=int(model_cfg["dim_feedforward"]),
            dropout_input=float(model_cfg.get("dropout_input", 0.0)),
            dropout_hidden=float(model_cfg.get("dropout_hidden", 0.0)),
            pooling=str(model_cfg.get("pooling", "last")),
        )
    raise ValueError(f"Unsupported model_name: {model_name}")


def train_single_run(config: dict[str, Any]) -> tuple[dict[str, Any], RunArtifacts]:
    config = deepcopy(config)
    model_name = str(config.get("model_name", "cnn"))
    dataset_name = str(config["dataset"]["name"])
    output_dir = config.get("output", {}).get("base_dir", "results/training")
    run_name = config.get("output", {}).get("run_name")
    run_dir = build_run_dir(output_dir, model_name, dataset_name, run_name=run_name)
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    seed = int(config.get("seed", 42))
    set_seed(seed)

    data_module = SequenceDataModule(
        dataset_name=dataset_name,
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["training"].get("num_workers", 0)),
        pin_memory=bool(config["training"].get("pin_memory", True)),
        include_test=bool(config["training"].get("evaluate_test", False)),
    )
    data_module.setup()
    assert data_module.bundle is not None

    model = build_model(model_name, config["model"], data_module.bundle.vocab_size, data_module.bundle.seq_len)
    lightning_module = LightningSequenceRegressor(model=model, training_config=config["training"], y_scaler=data_module.bundle.y_scaler)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoints_dir),
        filename="best",
        monitor="val_id_mae",
        mode="min",
        save_top_k=1,
    )
    early_stopping = EarlyStopping(
        monitor="val_id_mae",
        mode="min",
        patience=int(config["training"].get("early_stopping_patience", 20)),
    )

    mlflow_logger = create_mlflow_logger(config, model_name=model_name, dataset_name=dataset_name)
    if hasattr(mlflow_logger, "log_hyperparams"):
        mlflow_logger.log_hyperparams(flatten_dict(config))

    trainer = pl.Trainer(
        accelerator=config["training"].get("accelerator", "auto"),
        devices=config["training"].get("devices", "auto"),
        max_epochs=int(config["training"].get("epochs", 100)),
        log_every_n_steps=int(config["training"].get("log_every_n_steps", 10)),
        logger=None if isinstance(mlflow_logger, NullMlflowLogger) else mlflow_logger,
        callbacks=[checkpoint_callback, early_stopping],
        deterministic=bool(config["training"].get("deterministic", True)),
        enable_progress_bar=bool(config["training"].get("enable_progress_bar", True)),
        default_root_dir=str(run_dir),
    )

    trainer.fit(
        model=lightning_module,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=[data_module.val_id_dataloader(), data_module.val_ood_dataloader()],
    )

    best_ckpt_path = checkpoint_callback.best_model_path or None
    if best_ckpt_path:
        best_module = LightningSequenceRegressor.load_from_checkpoint(
            best_ckpt_path,
            model=model,
            training_config=config["training"],
            y_scaler=data_module.bundle.y_scaler,
        )
    else:
        best_module = lightning_module
    best_module = best_module.to(trainer.strategy.root_device)

    val_id_metrics = evaluate_regression_model(best_module, data_module.val_id_dataloader(), data_module.bundle.y_scaler)
    val_ood_metrics = evaluate_regression_model(best_module, data_module.val_ood_dataloader(), data_module.bundle.y_scaler)

    metrics = {
        "dataset": dataset_name,
        "model_name": model_name,
        "seed": seed,
        "seq_len": data_module.bundle.seq_len,
        "vocab_size": data_module.bundle.vocab_size,
        "best_checkpoint_monitor": "val_id_mae",
        "val_id": val_id_metrics,
        "val_ood": val_ood_metrics,
        "selected_metric": float(val_ood_metrics["mae"]),
    }
    if data_module.test_dataset is not None:
        metrics["test"] = evaluate_regression_model(best_module, data_module.test_dataloader(), data_module.bundle.y_scaler)

    model_state_dict_path = run_dir / "model_state_dict.pt"
    torch.save(best_module.model.state_dict(), model_state_dict_path)
    metrics_path = save_json(run_dir / "metrics.json", metrics)
    config_path = save_yaml(run_dir / "config.yaml", config)

    y_scaler_path = None
    if data_module.bundle.y_scaler is not None:
        y_scaler_path = save_json(run_dir / "y_scaler.json", data_module.bundle.y_scaler)

    mlflow_run_id_path = None
    run_id = getattr(mlflow_logger, "run_id", None)
    if run_id is not None:
        mlflow_run_id_path = run_dir / "mlflow_run_id.txt"
        mlflow_run_id_path.write_text(str(run_id), encoding="utf-8")

    log_metrics = {
        "val_id_mae": float(val_id_metrics["mae"]),
        "val_ood_mae": float(val_ood_metrics["mae"]),
        "val_id_std_abs_error": float(val_id_metrics["std_abs_error"]),
        "val_ood_std_abs_error": float(val_ood_metrics["std_abs_error"]),
    }
    if hasattr(mlflow_logger, "log_metrics"):
        mlflow_logger.log_metrics(log_metrics)

    artifacts = RunArtifacts(
        run_dir=run_dir,
        best_checkpoint_path=None if best_ckpt_path is None else Path(best_ckpt_path),
        model_state_dict_path=model_state_dict_path,
        metrics_path=metrics_path,
        config_path=config_path,
        y_scaler_path=y_scaler_path,
        mlflow_run_id_path=mlflow_run_id_path,
    )
    return metrics, artifacts


def train_one_run(
    base_config_path,
    override_config_paths=None,
    run_name=None,
    use_mlflow=True,
):
    config = load_config(base_config_path, override_config_paths[0] if override_config_paths else None)
    config.setdefault("output", {})
    if run_name is not None:
        config["output"]["run_name"] = run_name
    config.setdefault("mlflow", {})
    config["mlflow"]["enabled"] = use_mlflow
    metrics, artifacts = train_single_run(config)
    return {
        "metrics": metrics,
        "run_dir": artifacts.run_dir,
        "best_checkpoint_path": artifacts.best_checkpoint_path,
        "state_dict_path": artifacts.model_state_dict_path,
        "config_path": artifacts.config_path,
        "metrics_path": artifacts.metrics_path,
        "y_scaler_path": artifacts.y_scaler_path,
        "mlflow_run_id_path": artifacts.mlflow_run_id_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one deep learning model run with PyTorch Lightning.")
    parser.add_argument("--config", required=True, help="Path to base YAML config.")
    parser.add_argument("--override", default=None, help="Optional YAML override path.")
    args = parser.parse_args()
    config = load_config(args.config, args.override)
    metrics, artifacts = train_single_run(config)
    print(json.dumps({"metrics": metrics, "run_dir": str(artifacts.run_dir)}, indent=2))


if __name__ == "__main__":
    main()
