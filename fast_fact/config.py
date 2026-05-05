"""Dataclass configs for SEC fetching, dataset construction, and model training."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SECConfig:
    """Configuration for SEC EDGAR fetching.

    Attributes:
        user_agent: Descriptive User-Agent string with contact info, as
            required by the SEC. Format: "Name email@example.com".
        min_chars: Minimum character length for a filing's extracted text;
            shorter filings are dropped.
        item_filter: If True, only keep 8-Ks containing Item 2.02 (results
            of operations) or Item 4.02 (non-reliance on prior financials).
        max_filings_per_ticker: Cap on filings per ticker (oldest-first).
    """

    user_agent: str = "fast_fact contact@example.com"
    min_chars: int = 500
    item_filter: bool = True
    max_filings_per_ticker: int = 200


@dataclass
class DataConfig:
    """Configuration for the labeling/dataset pipeline.

    Attributes:
        tickers: Equity tickers to fetch 8-Ks for.
        start_date: Inclusive lower bound on filing_date (YYYY-MM-DD).
        end_date: Inclusive upper bound on filing_date (YYYY-MM-DD).
        horizon_days: Trading-day window over which to compute abnormal returns.
        label_threshold: Magnitude of abnormal return required to assign a
            +/- label; events below this magnitude are dropped as ambiguous.
    """

    tickers: List[str] = field(default_factory=list)
    start_date: str = "2011-01-01"
    end_date: str = "2024-12-31"
    horizon_days: int = 5
    label_threshold: float = 0.01


@dataclass
class ModelConfig:
    """Configuration for the text classifier and training loops."""

    base_model: str = "roberta-base"
    max_length: int = 1024
    batch_size: int = 8
    grad_accum_steps: int = 4
    num_epochs: int = 3
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    dpo_beta: float = 0.5
