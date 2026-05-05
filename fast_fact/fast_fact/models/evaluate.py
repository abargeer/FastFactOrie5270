"""Evaluation: classifier metrics, FinBERT prior, conflict slice, and a paper portfolio."""

from __future__ import annotations

from typing import Dict

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
from tqdm import tqdm


def evaluate_classifier(model, dataloader: DataLoader, device) -> Dict[str, float]:
    """Compute AUC / F1 / balanced-accuracy / Brier on a binary classifier."""
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].cpu().numpy()
            all_labels.append(labels)

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
            all_probs.append(probs.detach().cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_score = np.concatenate(all_probs)
    y_pred = (y_score >= 0.5).astype(int)

    metrics: Dict[str, float] = {}
    try:
        metrics["auc"] = roc_auc_score(y_true, y_score)
    except ValueError:
        metrics["auc"] = float("nan")
    metrics["f1"] = f1_score(y_true, y_pred)
    metrics["balanced_acc"] = balanced_accuracy_score(y_true, y_pred)
    metrics["brier"] = brier_score_loss(y_true, y_score)
    return metrics


def build_finbert_prior(
    df: pd.DataFrame,
    text_col: str,
    max_length: int = 512,
    confidence_thresh: float = 0.3,
    model_loader=None,
) -> pd.Series:
    """Run ProsusAI/finbert sentiment as an off-the-shelf directional prior.

    Maps positive/negative sentiment to labels 1/0, with neutral or
    low-confidence predictions returned as None. ``model_loader`` is a hook
    used by tests to inject a stub ``(tokenizer, model)`` factory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_loader is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        def _default_loader():
            tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            mdl = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            return tok, mdl

        model_loader = _default_loader

    try:
        tokenizer, model = model_loader()
    except Exception as e:
        print(f"[WARN] Could not load FinBERT prior: {e}")
        return pd.Series([None] * len(df), index=df.index)

    model.to(device)
    model.eval()

    priors = []
    confs = []

    for text in tqdm(df[text_col].tolist(), desc="FinBERT prior"):
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        p_neg, _p_neu, p_pos = probs.tolist()
        if p_pos > p_neg:
            prior, conf = 1, p_pos - p_neg
        elif p_neg > p_pos:
            prior, conf = 0, p_neg - p_pos
        else:
            prior, conf = None, 0.0
        if conf < confidence_thresh:
            prior = None
        priors.append(prior)
        confs.append(conf)

    df = df.copy()
    df["finbert_prior"] = priors
    df["finbert_conf"] = confs
    return df["finbert_prior"]


def conflict_slice_mask(
    df: pd.DataFrame,
    prior_col: str = "finbert_prior",
    label_col: str = "label",
) -> pd.Series:
    """Boolean mask of rows where a defined prior disagrees with the realized label."""
    prior = pd.to_numeric(df[prior_col], errors="coerce")
    labels = pd.to_numeric(df[label_col], errors="coerce")
    valid = prior.notna() & labels.notna()
    return valid & (prior != labels)


def portfolio_sanity_check(
    df: pd.DataFrame,
    prob_col: str = "p_up",
    abn_ret_col: str = "abn_ret",
    cut: float = 0.5,
) -> Dict[str, float]:
    """Equal-weight long/short paper portfolio summary.

    * Long if predicted up-probability >= ``cut``.
    * Short if predicted up-probability <= 1 - ``cut``.
    * Flat otherwise.

    Returns mean per-event abnormal returns for each leg and the combined
    portfolio, along with leg counts.
    """
    p = df[prob_col].values
    r = df[abn_ret_col].values

    long_mask = p >= cut
    short_mask = p <= (1.0 - cut)

    long_ret = r[long_mask].mean() if long_mask.any() else 0.0
    short_ret = -r[short_mask].mean() if short_mask.any() else 0.0
    if long_mask.any() or short_mask.any():
        combined = np.concatenate([r[long_mask], -r[short_mask]])
    else:
        combined = np.array([0.0])
    total_ret = combined.mean()

    return {
        "mean_long_ret": float(long_ret),
        "mean_short_ret": float(short_ret),
        "mean_total_ret": float(total_ret),
        "num_long": int(long_mask.sum()),
        "num_short": int(short_mask.sum()),
    }
