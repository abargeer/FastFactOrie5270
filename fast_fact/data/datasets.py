"""PyTorch Dataset wrappers for supervised and DPO-style training."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import Dataset


class TextClassificationDataset(Dataset):
    """Tokenized text classification examples for supervised training."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        text_col: str,
        max_length: int,
    ):
        self.texts: List[str] = df[text_col].tolist()
        self.labels: List[int] = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class DPODataset(Dataset):
    """Pairwise dataset for DPO-style preference training.

    For each (text, label y) we treat y as the "chosen" class and 1-y as
    "rejected"; the DPO objective compares log p(y|x) vs log p(1-y|x) for the
    policy and the frozen reference.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        text_col: str,
        max_length: int,
    ):
        self.texts: List[str] = df[text_col].tolist()
        self.labels: List[int] = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        y = self.labels[idx]
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["chosen_labels"] = torch.tensor(y, dtype=torch.long)
        item["rejected_labels"] = torch.tensor(1 - y, dtype=torch.long)
        return item


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack a list of per-example tensor dicts along dim 0."""
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
