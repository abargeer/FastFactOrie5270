"""Data collection, labeling, splitting, and PyTorch dataset utilities."""

from fast_fact.data.datasets import (
    DPODataset,
    TextClassificationDataset,
    collate_fn,
)
from fast_fact.data.prices import (
    attach_abnormal_returns_and_labels,
    compute_beta,
    download_price_history,
)
from fast_fact.data.sec import (
    collect_8k_events_for_universe,
    extract_text_from_filing,
    fetch_company_submissions,
    load_ticker_to_cik_map,
)
from fast_fact.data.splits import add_rag_context, make_time_splits

__all__ = [
    "DPODataset",
    "TextClassificationDataset",
    "add_rag_context",
    "attach_abnormal_returns_and_labels",
    "collate_fn",
    "collect_8k_events_for_universe",
    "compute_beta",
    "download_price_history",
    "extract_text_from_filing",
    "fetch_company_submissions",
    "load_ticker_to_cik_map",
    "make_time_splits",
]
