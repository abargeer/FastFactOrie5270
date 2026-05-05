"""Model construction, training (SFT/RAG/DPO), and evaluation utilities."""

from fast_fact.models.base import apply_lora, build_base_model, freeze_encoder
from fast_fact.models.evaluate import (
    build_finbert_prior,
    conflict_slice_mask,
    evaluate_classifier,
    portfolio_sanity_check,
)
from fast_fact.models.train import (
    train_dpo,
    train_rag_baseline,
    train_supervised_lora,
)

__all__ = [
    "apply_lora",
    "build_base_model",
    "build_finbert_prior",
    "conflict_slice_mask",
    "evaluate_classifier",
    "freeze_encoder",
    "portfolio_sanity_check",
    "train_dpo",
    "train_rag_baseline",
    "train_supervised_lora",
]
