from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ssl_for_ood.data.load_data import _get_data_root
from ssl_for_ood.data.preprocess_data import DATASET_TYPE, load_preprocessed_split, load_y_scaler


VOCAB_SIZE_BY_DATASET_TYPE = {
    "dna": 4,
    "protein": 20,
}


class SequenceRegressionDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray | None) -> None:
        self.x = torch.as_tensor(x, dtype=torch.long)
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int):
        if self.y is None:
            return self.x[index]
        return self.x[index], self.y[index]


@dataclass
class DatasetBundle:
    dataset_name: str
    dataset_type: str
    vocab_size: int
    seq_len: int
    y_scaler: dict[str, float] | None
    train: dict[str, Any]
    val_id: dict[str, Any]
    val_ood: dict[str, Any]
    test: dict[str, Any] | None = None
    target_unlabeled: dict[str, Any] | None = None


class SequenceDataModule:
    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = True,
        include_test: bool = False,
        include_target_unlabeled: bool = False,
    ) -> None:
        if dataset_name not in DATASET_TYPE:
            valid = ", ".join(sorted(DATASET_TYPE))
            raise ValueError(f"Unknown dataset '{dataset_name}'. Expected one of: {valid}")
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.include_test = include_test
        self.include_target_unlabeled = include_target_unlabeled
        self.bundle: DatasetBundle | None = None
        self.train_dataset: SequenceRegressionDataset | None = None
        self.val_id_dataset: SequenceRegressionDataset | None = None
        self.val_ood_dataset: SequenceRegressionDataset | None = None
        self.test_dataset: SequenceRegressionDataset | None = None
        self.target_unlabeled_dataset: SequenceRegressionDataset | None = None

    def setup(self) -> None:
        dataset_type = DATASET_TYPE[self.dataset_name]
        vocab_size = VOCAB_SIZE_BY_DATASET_TYPE[dataset_type]
        train = load_preprocessed_split(self.dataset_name, "train")
        val_id = load_preprocessed_split(self.dataset_name, "val_id")
        val_ood = load_preprocessed_split(self.dataset_name, "val_ood")
        test = load_preprocessed_split(self.dataset_name, "test") if self.include_test else None
        target_unlabeled = (
            load_preprocessed_split(self.dataset_name, "target_unlabeled") if self.include_target_unlabeled else None
        )
        try:
            y_scaler = load_y_scaler(self.dataset_name)
        except FileNotFoundError:
            y_scaler = None
        self.bundle = DatasetBundle(
            dataset_name=self.dataset_name,
            dataset_type=dataset_type,
            vocab_size=vocab_size,
            seq_len=int(train["seq_len"]),
            y_scaler=y_scaler,
            train=train,
            val_id=val_id,
            val_ood=val_ood,
            test=test,
            target_unlabeled=target_unlabeled,
        )
        self.train_dataset = SequenceRegressionDataset(train["x"], train["y"])
        self.val_id_dataset = SequenceRegressionDataset(val_id["x"], val_id["y"])
        self.val_ood_dataset = SequenceRegressionDataset(val_ood["x"], val_ood["y"])
        self.test_dataset = None if test is None else SequenceRegressionDataset(test["x"], test["y"])
        self.target_unlabeled_dataset = None if target_unlabeled is None else SequenceRegressionDataset(target_unlabeled["x"], target_unlabeled["y"])

    def _loader(self, dataset: SequenceRegressionDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return self._loader(self.train_dataset, shuffle=True)

    def val_id_dataloader(self) -> DataLoader:
        if self.val_id_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return self._loader(self.val_id_dataset, shuffle=False)

    def val_ood_dataloader(self) -> DataLoader:
        if self.val_ood_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return self._loader(self.val_ood_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Test split was not loaded.")
        return self._loader(self.test_dataset, shuffle=False)


def get_data_root() -> Path:
    return _get_data_root()
