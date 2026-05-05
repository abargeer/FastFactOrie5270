"""Shared test fixtures: a tiny tokenizer, a tiny classifier model, and labeled events."""

from __future__ import annotations

from types import SimpleNamespace
from typing import List

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyTokenizer:
    """Char-level tokenizer that returns torch tensors shaped like HF outputs."""

    def __init__(self, vocab_size: int = 64):
        self.vocab_size = vocab_size

    def __call__(
        self,
        text,
        truncation: bool = True,
        max_length: int = 16,
        padding: str = "max_length",
        return_tensors: str = "pt",
    ):
        if isinstance(text, str):
            texts: List[str] = [text]
            single = True
        else:
            texts = list(text)
            single = False
        ids_list = []
        mask_list = []
        for t in texts:
            ids = [ord(c) % self.vocab_size for c in t][:max_length]
            mask = [1] * len(ids)
            pad = max_length - len(ids)
            if pad > 0:
                ids += [0] * pad
                mask += [0] * pad
            ids_list.append(ids)
            mask_list.append(mask)
        out = {
            "input_ids": torch.tensor(ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(mask_list, dtype=torch.long),
        }
        if single and return_tensors == "pt":
            return {k: v for k, v in out.items()}  # keeps [1, L] shape
        return out


class TinyClassifier(nn.Module):
    """Minimal HF-shaped classifier: embedding + mean-pool + linear head.

    Exposes a ``roberta`` attribute so ``freeze_encoder`` can find an encoder
    to freeze.
    """

    def __init__(self, vocab_size: int = 64, hidden: int = 4, num_labels: int = 2):
        super().__init__()
        self.roberta = nn.Embedding(vocab_size, hidden)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.roberta(input_ids).mean(dim=1)
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return SimpleNamespace(logits=logits, loss=loss)


@pytest.fixture
def tiny_tokenizer():
    return TinyTokenizer()


@pytest.fixture
def tiny_classifier_factory():
    def _make():
        torch.manual_seed(0)
        return TinyClassifier()

    return _make


@pytest.fixture
def labeled_events_df() -> pd.DataFrame:
    """Synthetic labeled events DataFrame with all columns the pipeline needs."""
    rng = np.random.default_rng(42)
    n = 24
    accepted = pd.date_range("2020-01-01", periods=n, freq="7D")
    df = pd.DataFrame(
        {
            "event_id": [f"e{i}" for i in range(n)],
            "ticker": (["AAA"] * (n // 2)) + (["BBB"] * (n - n // 2)),
            "cik": ["0000000001"] * (n // 2) + ["0000000002"] * (n - n // 2),
            "form": ["8-K"] * n,
            "filing_date": accepted,
            "accepted_dt": accepted,
            "primary_doc": [f"doc{i}.htm" for i in range(n)],
            "filing_url": [f"http://example.com/{i}" for i in range(n)],
            "text": [f"filing text number {i} item 2.02" for i in range(n)],
            "entry_date": accepted,
            "exit_date": accepted + pd.Timedelta(days=5),
            "abn_ret": rng.normal(0, 0.02, size=n),
            "label": rng.integers(0, 2, size=n).tolist(),
            "label_end_dt": accepted + pd.Timedelta(days=5),
        }
    )
    return df
