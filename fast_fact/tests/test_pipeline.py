"""Tests for fast_fact.pipeline.

The full ``run_experiment`` is heavy, so we test the small helpers directly
and stub the trainer factories for the smoke run.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from fast_fact import pipeline as pipeline_mod
from fast_fact.config import DataConfig, ModelConfig, SECConfig


def test_prepare_events_uses_existing_rag_cache(tmp_path):
    """If the RAG parquet already exists, prepare_events just reads it."""
    df = pd.DataFrame({
        "ticker": ["AAA", "AAA"],
        "accepted_dt": pd.to_datetime(["2020-01-01", "2020-01-08"]),
        "filing_date": pd.to_datetime(["2020-01-01", "2020-01-08"]),
        "text": ["a", "b"],
        "rag_text": ["a", "b"],
        "label": [0, 1],
        "label_end_dt": pd.to_datetime(["2020-01-06", "2020-01-13"]),
        "abn_ret": [-0.01, 0.02],
    })
    df.to_parquet(tmp_path / "events_with_rag.parquet")
    out = pipeline_mod.prepare_events(
        DataConfig(tickers=["AAA"]), SECConfig(), str(tmp_path), use_cache=True
    )
    assert len(out) == 2


def test_prepare_events_pipes_through_when_only_labeled_exists(tmp_path):
    df = pd.DataFrame({
        "ticker": ["AAA", "AAA", "AAA"],
        "accepted_dt": pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-15"]),
        "filing_date": pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-15"]),
        "text": ["a", "b", "c"],
        "label": [0, 1, 0],
        "label_end_dt": pd.to_datetime(["2020-01-06", "2020-01-13", "2020-01-20"]),
        "abn_ret": [-0.01, 0.02, -0.03],
    })
    df.to_parquet(tmp_path / "events_labeled.parquet")
    out = pipeline_mod.prepare_events(
        DataConfig(tickers=["AAA"]), SECConfig(), str(tmp_path), use_cache=True
    )
    assert "rag_text" in out.columns
    assert (tmp_path / "events_with_rag.parquet").exists()


def test_prepare_events_full_path(monkeypatch, tmp_path):
    """Both caches missing -> the SEC + price helpers are called."""
    raw = pd.DataFrame({
        "ticker": ["AAA"], "accepted_dt": [pd.Timestamp("2020-01-15")],
        "filing_date": [pd.Timestamp("2020-01-15")], "text": ["x"],
    })
    labeled = raw.copy()
    labeled["label"] = [1]
    labeled["label_end_dt"] = [pd.Timestamp("2020-01-22")]
    labeled["abn_ret"] = [0.01]

    monkeypatch.setattr(pipeline_mod, "collect_8k_events_for_universe",
                        lambda dc, sc: raw)
    monkeypatch.setattr(pipeline_mod, "download_price_history",
                        lambda *a, **kw: {})
    monkeypatch.setattr(pipeline_mod, "attach_abnormal_returns_and_labels",
                        lambda events, prices, cfg: labeled)

    out = pipeline_mod.prepare_events(
        DataConfig(tickers=["AAA"]), SECConfig(), str(tmp_path), use_cache=True
    )
    assert (tmp_path / "events_raw.parquet").exists()
    assert (tmp_path / "events_labeled.parquet").exists()
    assert (tmp_path / "events_with_rag.parquet").exists()
    assert "rag_text" in out.columns


def test_make_datasets_returns_all_seven(labeled_events_df, tiny_tokenizer):
    df = labeled_events_df.copy()
    df["rag_text"] = df["text"]
    train, val, test = df.iloc[:8], df.iloc[8:14], df.iloc[14:]
    cfg = ModelConfig(max_length=8)
    ds = pipeline_mod.make_datasets(train, val, test, tiny_tokenizer, cfg)
    assert set(ds.keys()) == {
        "train_raw", "val_raw", "test_raw",
        "train_rag", "val_rag", "test_rag",
        "train_dpo",
    }
    for k in ds:
        assert len(ds[k]) > 0


def test_predict_p_up(labeled_events_df, tiny_tokenizer, tiny_classifier_factory):
    from torch.utils.data import DataLoader

    from fast_fact.data.datasets import TextClassificationDataset, collate_fn

    ds = TextClassificationDataset(labeled_events_df, tiny_tokenizer, "text", max_length=8)
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_fn)
    probs = pipeline_mod._predict_p_up(
        tiny_classifier_factory(), loader, torch.device("cpu")
    )
    assert probs.shape == (len(labeled_events_df),)
    assert ((0.0 <= probs) & (probs <= 1.0)).all()


def test_prior_metrics_returns_dict_and_conflict(labeled_events_df):
    df = labeled_events_df.copy()
    df["p_up_rag"] = 0.5
    df["p_up_lora"] = 0.5
    df["p_up_dpo"] = 0.5
    p = np.where(df["label"] == 1, 0.05, 0.95)  # confidently wrong on every row
    metrics, conflict = pipeline_mod._prior_metrics(df, p)
    assert "auc" in metrics
    assert len(conflict) > 0


def test_conflict_metrics_panel(labeled_events_df):
    df = labeled_events_df.copy()
    for col in ["p_up_prior", "p_up_rag", "p_up_lora", "p_up_dpo"]:
        df[col] = np.linspace(0.1, 0.9, len(df))
    out = pipeline_mod._conflict_metrics(df)
    assert set(out.keys()) == {"prior", "rag", "lora", "dpo"}
    for v in out.values():
        assert {"auc", "f1", "balanced_acc", "brier"} <= set(v.keys())


def test_run_experiment_smoke(monkeypatch, tmp_path, labeled_events_df,
                              tiny_tokenizer, tiny_classifier_factory):
    """Exercise run_experiment end-to-end with stubbed trainers."""
    df = labeled_events_df.copy()
    df["rag_text"] = df["text"]
    df.to_parquet(tmp_path / "events_with_rag.parquet")

    from fast_fact.data import splits as splits_mod

    def stub_splits(events, train_q=0.6, val_q=0.8):
        return events.iloc[:10].reset_index(drop=True), \
               events.iloc[10:16].reset_index(drop=True), \
               events.iloc[16:].reset_index(drop=True)

    monkeypatch.setattr(pipeline_mod, "make_time_splits", stub_splits)
    monkeypatch.setattr(splits_mod, "make_time_splits", stub_splits)

    monkeypatch.setattr(pipeline_mod, "train_rag_baseline",
                        lambda *a, **kw: tiny_classifier_factory())
    monkeypatch.setattr(pipeline_mod, "train_supervised_lora",
                        lambda *a, **kw: tiny_classifier_factory())
    monkeypatch.setattr(pipeline_mod, "train_dpo",
                        lambda *a, **kw: tiny_classifier_factory())
    monkeypatch.setattr(pipeline_mod, "build_base_model",
                        lambda cfg, num_labels=2: tiny_classifier_factory())

    cfg = ModelConfig(base_model="dummy", max_length=8, batch_size=2, num_epochs=1)
    summary = pipeline_mod.run_experiment(
        DataConfig(tickers=["AAA"]),
        SECConfig(),
        cfg,
        outdir=str(tmp_path),
        use_cache=True,
        tokenizer=tiny_tokenizer,
    )
    assert "rag_test" in summary
    assert (tmp_path / "summary.json").exists()
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved.keys() == summary.keys()
