"""Tests for the SFT/RAG/DPO training loops.

The HF/peft model factories are monkey-patched to return our TinyClassifier
so we don't pull a real RoBERTa down. The transformers cosine scheduler is
real (it's already a test dependency via the package install).
"""

from __future__ import annotations

import os

import pandas as pd
import pytest
import torch

from fast_fact.config import ModelConfig
from fast_fact.data.datasets import (
    DPODataset,
    TextClassificationDataset,
)
from fast_fact.models import train as train_mod


@pytest.fixture
def small_cfg():
    return ModelConfig(
        base_model="dummy",
        max_length=8,
        batch_size=2,
        grad_accum_steps=1,
        num_epochs=1,
        warmup_steps=0,
        lr=1e-3,
    )


def _make_train_val(labeled_events_df, tiny_tokenizer):
    train_df = labeled_events_df.iloc[:8].reset_index(drop=True)
    val_df = labeled_events_df.iloc[8:14].reset_index(drop=True)
    if not (set(train_df["label"]) >= {0, 1}):
        train_df.loc[0, "label"] = 0
        train_df.loc[1, "label"] = 1
    if not (set(val_df["label"]) >= {0, 1}):
        val_df.loc[0, "label"] = 0
        val_df.loc[1, "label"] = 1
    train_ds = TextClassificationDataset(train_df, tiny_tokenizer, "text", max_length=8)
    val_ds = TextClassificationDataset(val_df, tiny_tokenizer, "text", max_length=8)
    return train_ds, val_ds, train_df, val_df


def test_train_supervised_lora_runs_and_saves_checkpoint(
    monkeypatch, tmp_path, labeled_events_df, tiny_tokenizer,
    tiny_classifier_factory, small_cfg,
):
    monkeypatch.setattr(train_mod, "build_base_model",
                        lambda cfg, num_labels=2: tiny_classifier_factory())
    monkeypatch.setattr(train_mod, "apply_lora", lambda model, cfg: model)

    train_ds, val_ds, *_ = _make_train_val(labeled_events_df, tiny_tokenizer)
    out = train_mod.train_supervised_lora(
        train_ds, val_ds, tiny_tokenizer, small_cfg,
        output_dir=str(tmp_path), device=torch.device("cpu"),
    )
    assert out is not None
    assert os.path.exists(tmp_path / "best_lora.pt")


def test_train_rag_baseline_freezes_encoder_and_runs(
    monkeypatch, tmp_path, labeled_events_df, tiny_tokenizer,
    tiny_classifier_factory, small_cfg,
):
    captured = {}

    def fake_freeze(model):
        for p in model.roberta.parameters():
            p.requires_grad = False
        captured["frozen"] = True
        return model

    monkeypatch.setattr(train_mod, "build_base_model",
                        lambda cfg, num_labels=2: tiny_classifier_factory())
    monkeypatch.setattr(train_mod, "freeze_encoder", fake_freeze)

    train_ds, val_ds, *_ = _make_train_val(labeled_events_df, tiny_tokenizer)
    out = train_mod.train_rag_baseline(
        train_ds, val_ds, tiny_tokenizer, small_cfg,
        output_dir=str(tmp_path), device=torch.device("cpu"),
    )
    assert captured["frozen"] is True
    assert out is not None
    assert os.path.exists(tmp_path / "best_rag.pt")


def test_train_dpo_runs(
    monkeypatch, tmp_path, labeled_events_df, tiny_tokenizer,
    tiny_classifier_factory, small_cfg,
):
    monkeypatch.setattr(train_mod, "build_base_model",
                        lambda cfg, num_labels=2: tiny_classifier_factory())
    monkeypatch.setattr(train_mod, "apply_lora", lambda model, cfg: model)

    train_df = labeled_events_df.iloc[:8].reset_index(drop=True)
    val_df = labeled_events_df.iloc[8:14].reset_index(drop=True)
    if not (set(train_df["label"]) >= {0, 1}):
        train_df.loc[0, "label"] = 0
        train_df.loc[1, "label"] = 1
    if not (set(val_df["label"]) >= {0, 1}):
        val_df.loc[0, "label"] = 0
        val_df.loc[1, "label"] = 1
    train_ds = DPODataset(train_df, tiny_tokenizer, "text", max_length=8)
    val_ds = TextClassificationDataset(val_df, tiny_tokenizer, "text", max_length=8)

    sft = tiny_classifier_factory()
    out = train_mod.train_dpo(
        train_ds, val_ds, tiny_tokenizer, small_cfg,
        lora_sft_model=sft, output_dir=str(tmp_path), device=torch.device("cpu"),
    )
    assert out is not None
    assert os.path.exists(tmp_path / "best_dpo.pt")


def test_make_loaders_helper(labeled_events_df, tiny_tokenizer):
    train_ds, val_ds, *_ = _make_train_val(labeled_events_df, tiny_tokenizer)
    train_loader, val_loader = train_mod._make_loaders(train_ds, val_ds, batch_size=2)
    assert hasattr(train_loader, "__iter__")
    assert hasattr(val_loader, "__iter__")
