"""Tests for fast_fact.models.evaluate — uses TinyClassifier instead of HF models."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from fast_fact.data.datasets import TextClassificationDataset, collate_fn
from fast_fact.models import evaluate as eval_mod


def test_evaluate_classifier_returns_metric_dict(
    labeled_events_df, tiny_tokenizer, tiny_classifier_factory
):
    ds = TextClassificationDataset(labeled_events_df, tiny_tokenizer, "text", max_length=8)
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_fn)
    model = tiny_classifier_factory()
    metrics = eval_mod.evaluate_classifier(model, loader, torch.device("cpu"))
    for key in ("auc", "f1", "balanced_acc", "brier"):
        assert key in metrics


def test_evaluate_classifier_handles_single_class():
    """If all labels are the same, AUC is undefined and should fall back to NaN."""
    df = pd.DataFrame({
        "text": ["a"] * 4,
        "label": [1, 1, 1, 1],
    })

    class StubTok:
        def __call__(self, text, **kw):
            return {
                "input_ids": torch.zeros(1, 4, dtype=torch.long),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
            }

    ds = TextClassificationDataset(df, StubTok(), "text", max_length=4)
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_fn)

    class FixedModel(torch.nn.Module):
        def forward(self, input_ids, attention_mask=None):
            return SimpleNamespace(logits=torch.zeros(input_ids.shape[0], 2))

    metrics = eval_mod.evaluate_classifier(FixedModel(), loader, torch.device("cpu"))
    assert np.isnan(metrics["auc"])  # one-class AUC is NaN


def test_conflict_slice_mask():
    df = pd.DataFrame({
        "finbert_prior": [0, 1, None, 1, 0],
        "label":         [1, 1, 1,    0, 0],
    })
    mask = eval_mod.conflict_slice_mask(df)
    # row 0: 0 != 1 -> True; row 1: 1 == 1 -> False; row 2: None -> False;
    # row 3: 1 != 0 -> True; row 4: 0 == 0 -> False
    assert mask.tolist() == [True, False, False, True, False]


def test_portfolio_sanity_check_long_short_and_flat():
    df = pd.DataFrame({
        "p_up":    [0.9, 0.1, 0.5, 0.95, 0.05],
        "abn_ret": [0.01, -0.02, 0.0, 0.03, -0.04],
    })
    out = eval_mod.portfolio_sanity_check(df, "p_up", "abn_ret", cut=0.6)
    assert out["num_long"] == 2  # 0.9, 0.95
    assert out["num_short"] == 2  # 0.1, 0.05
    assert out["mean_long_ret"] == pytest.approx(np.mean([0.01, 0.03]))
    assert out["mean_short_ret"] == pytest.approx(-np.mean([-0.02, -0.04]))


def test_portfolio_sanity_check_all_flat():
    df = pd.DataFrame({"p_up": [0.5, 0.5], "abn_ret": [0.0, 0.0]})
    out = eval_mod.portfolio_sanity_check(df, "p_up", "abn_ret", cut=0.9)
    assert out["num_long"] == 0
    assert out["num_short"] == 0


def test_build_finbert_prior_with_injected_loader():
    df = pd.DataFrame({"text": ["pos sentence", "neg sentence", "neutral"]})

    class StubTokenizer:
        def __call__(self, text, **kw):
            return {
                "input_ids": torch.zeros(1, 4, dtype=torch.long),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
            }

    class StubFinbert(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def __call__(self, **kw):  # mimic HF call returning .logits
            i = self.calls
            self.calls += 1
            # Order: row 0 -> positive (high p_pos), 1 -> negative, 2 -> neutral (low confidence)
            tables = [
                torch.tensor([[-5.0, -5.0, 5.0]]),   # strongly positive
                torch.tensor([[5.0, -5.0, -5.0]]),   # strongly negative
                torch.tensor([[0.0, 0.0, 0.0]]),     # neutral
            ]
            return SimpleNamespace(logits=tables[i % len(tables)])

        def to(self, device):
            return self

        def eval(self):
            return self

    out = eval_mod.build_finbert_prior(
        df, "text",
        max_length=4, confidence_thresh=0.3,
        model_loader=lambda: (StubTokenizer(), StubFinbert()),
    )
    assert out.iloc[0] == 1
    assert out.iloc[1] == 0
    assert pd.isna(out.iloc[2])  # below confidence threshold -> stored as NaN


def test_build_finbert_prior_loader_failure_returns_none_series():
    df = pd.DataFrame({"text": ["a", "b"]})

    def boom():
        raise RuntimeError("hf hub down")

    out = eval_mod.build_finbert_prior(df, "text", model_loader=boom)
    assert out.isna().all()
    assert len(out) == 2


