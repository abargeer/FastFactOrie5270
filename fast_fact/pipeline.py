"""End-to-end experiment orchestration.

The pipeline is split into small, individually testable steps:

1. :func:`prepare_events` - collect 8-Ks, label them, attach RAG context.
   Each step writes a parquet so re-runs can resume without re-fetching.
2. :func:`make_datasets` - build PyTorch Datasets from a labeled DataFrame.
3. :func:`run_experiment` - glues steps 1-2 together, then trains and
   evaluates the RAG, LoRA-SFT, and DPO models, plus an untuned base prior.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from fast_fact.config import DataConfig, ModelConfig, SECConfig
from fast_fact.data.datasets import (
    DPODataset,
    TextClassificationDataset,
    collate_fn,
)
from fast_fact.data.prices import (
    attach_abnormal_returns_and_labels,
    download_price_history,
)
from fast_fact.data.sec import collect_8k_events_for_universe
from fast_fact.data.splits import add_rag_context, make_time_splits
from fast_fact.models.base import build_base_model
from fast_fact.models.evaluate import evaluate_classifier, portfolio_sanity_check
from fast_fact.models.train import (
    train_dpo,
    train_rag_baseline,
    train_supervised_lora,
)


def prepare_events(
    data_cfg: DataConfig,
    sec_cfg: SECConfig,
    outdir: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Collect, label, and RAG-augment 8-K events; cache each stage to parquet."""
    os.makedirs(outdir, exist_ok=True)
    raw_path = os.path.join(outdir, "events_raw.parquet")
    labeled_path = os.path.join(outdir, "events_labeled.parquet")
    rag_path = os.path.join(outdir, "events_with_rag.parquet")

    if use_cache and os.path.exists(rag_path):
        return pd.read_parquet(rag_path)

    if use_cache and os.path.exists(labeled_path):
        events_labeled = pd.read_parquet(labeled_path)
    else:
        if use_cache and os.path.exists(raw_path):
            events = pd.read_parquet(raw_path)
        else:
            events = collect_8k_events_for_universe(data_cfg, sec_cfg)
            events.to_parquet(raw_path)

        prices = download_price_history(
            data_cfg.tickers, data_cfg.start_date, data_cfg.end_date
        )
        events_labeled = attach_abnormal_returns_and_labels(events, prices, data_cfg)
        events_labeled.to_parquet(labeled_path)

    events_with_rag = add_rag_context(events_labeled)
    events_with_rag.to_parquet(rag_path)
    return events_with_rag


def make_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer,
    model_cfg: ModelConfig,
) -> Dict[str, object]:
    """Construct the six classification datasets and one DPO dataset."""
    return {
        "train_raw": TextClassificationDataset(
            train_df, tokenizer, "text", model_cfg.max_length
        ),
        "val_raw": TextClassificationDataset(
            val_df, tokenizer, "text", model_cfg.max_length
        ),
        "test_raw": TextClassificationDataset(
            test_df, tokenizer, "text", model_cfg.max_length
        ),
        "train_rag": TextClassificationDataset(
            train_df, tokenizer, "rag_text", model_cfg.max_length
        ),
        "val_rag": TextClassificationDataset(
            val_df, tokenizer, "rag_text", model_cfg.max_length
        ),
        "test_rag": TextClassificationDataset(
            test_df, tokenizer, "rag_text", model_cfg.max_length
        ),
        "train_dpo": DPODataset(train_df, tokenizer, "text", model_cfg.max_length),
    }


def _predict_p_up(model, dataloader: DataLoader, device) -> np.ndarray:
    """Stack the per-batch P(label=1) over a dataloader."""
    model.eval()
    parts = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits
            parts.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(parts) if parts else np.array([])


def _prior_metrics(
    test_df: pd.DataFrame, p_up_prior: np.ndarray
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Score an untuned base model and pull out its high-confidence-wrong slice."""
    y_true = test_df["label"].astype(int).values
    y_pred = (p_up_prior >= 0.5).astype(int)
    try:
        prior_auc = roc_auc_score(y_true, p_up_prior)
    except ValueError:
        prior_auc = float("nan")
    metrics = {
        "auc": prior_auc,
        "f1": f1_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "brier": brier_score_loss(y_true, p_up_prior),
    }
    test_df = test_df.copy()
    test_df["p_up_prior"] = p_up_prior
    test_df["prior_pred"] = y_pred
    test_df["prior_conf"] = np.abs(p_up_prior - 0.5) * 2
    conflict_mask = (test_df["prior_conf"] >= 0.3) & (
        test_df["prior_pred"] != test_df["label"].astype(int)
    )
    return metrics, test_df[conflict_mask].reset_index(drop=True)


def _conflict_metrics(conflict_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute the four-model metric panel on the conflict slice."""

    def _on(prob_col: str) -> Dict[str, float]:
        y_true = conflict_df["label"].astype(int).values
        y_score = conflict_df[prob_col].values
        y_pred = (y_score >= 0.5).astype(int)
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = float("nan")
        return {
            "auc": auc,
            "f1": f1_score(y_true, y_pred),
            "balanced_acc": balanced_accuracy_score(y_true, y_pred),
            "brier": brier_score_loss(y_true, y_score),
        }

    return {name: _on(f"p_up_{name}") for name in ["prior", "rag", "lora", "dpo"]}


def run_experiment(
    data_cfg: DataConfig,
    sec_cfg: SECConfig,
    model_cfg: ModelConfig,
    outdir: str = "fast_fact_outputs",
    use_cache: bool = True,
    tokenizer=None,
) -> Dict:
    """Train and evaluate the RAG / LoRA / DPO trio plus an untuned base prior.

    Writes the full metric/portfolio summary to ``outdir/summary.json`` and
    returns the same dict to the caller.
    """
    from transformers import AutoTokenizer

    os.makedirs(outdir, exist_ok=True)
    events_with_rag = prepare_events(data_cfg, sec_cfg, outdir, use_cache=use_cache)
    train_df, val_df, test_df = make_time_splits(events_with_rag)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_cfg.base_model)
    ds = make_datasets(train_df, val_df, test_df, tokenizer, model_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n=== Training RAG baseline (frozen encoder + context) ===")
    rag_model = train_rag_baseline(
        ds["train_rag"], ds["val_rag"], tokenizer, model_cfg,
        output_dir=os.path.join(outdir, "rag"),
    )
    test_rag_loader = DataLoader(
        ds["test_rag"], batch_size=model_cfg.batch_size, shuffle=False, collate_fn=collate_fn,
    )
    rag_metrics = evaluate_classifier(rag_model, test_rag_loader, device)
    test_df = test_df.copy()
    test_df["p_up_rag"] = _predict_p_up(rag_model, test_rag_loader, device)
    rag_portfolio = portfolio_sanity_check(test_df, "p_up_rag", "abn_ret")

    print("\n=== Training LoRA SFT (no context) ===")
    lora_model = train_supervised_lora(
        ds["train_raw"], ds["val_raw"], tokenizer, model_cfg,
        output_dir=os.path.join(outdir, "lora"),
    )
    test_raw_loader = DataLoader(
        ds["test_raw"], batch_size=model_cfg.batch_size, shuffle=False, collate_fn=collate_fn,
    )
    lora_metrics = evaluate_classifier(lora_model, test_raw_loader, device)
    test_df["p_up_lora"] = _predict_p_up(lora_model, test_raw_loader, device)
    lora_portfolio = portfolio_sanity_check(test_df, "p_up_lora", "abn_ret")

    print("\n=== Training DPO-style model on top of LoRA SFT ===")
    dpo_model = train_dpo(
        ds["train_dpo"], ds["val_raw"], tokenizer, model_cfg,
        lora_sft_model=lora_model, output_dir=os.path.join(outdir, "dpo"),
    )
    dpo_metrics = evaluate_classifier(dpo_model, test_raw_loader, device)
    test_df["p_up_dpo"] = _predict_p_up(dpo_model, test_raw_loader, device)
    dpo_portfolio = portfolio_sanity_check(test_df, "p_up_dpo", "abn_ret")

    print("\n=== Evaluating base (untuned) prior on test set ===")
    prior_model = build_base_model(model_cfg, num_labels=2).to(device)
    prior_loader = DataLoader(
        ds["test_raw"], batch_size=model_cfg.batch_size, shuffle=False, collate_fn=collate_fn,
    )
    p_up_prior = _predict_p_up(prior_model, prior_loader, device)
    prior_metrics, conflict_df = _prior_metrics(test_df, p_up_prior)
    test_df["p_up_prior"] = p_up_prior

    conflict_metrics: Dict[str, Dict[str, float]] = {}
    if len(conflict_df) > 0:
        conflict_metrics = _conflict_metrics(conflict_df)
        conflict_df.to_parquet(os.path.join(outdir, "test_conflict_slice.parquet"))

    summary = {
        "prior_test": prior_metrics,
        "rag_test": rag_metrics,
        "lora_test": lora_metrics,
        "dpo_test": dpo_metrics,
        "rag_portfolio": rag_portfolio,
        "lora_portfolio": lora_portfolio,
        "dpo_portfolio": dpo_portfolio,
        "conflict_metrics": conflict_metrics,
    }
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== Experiment summary ===")
    print(json.dumps(summary, indent=2))
    return summary
