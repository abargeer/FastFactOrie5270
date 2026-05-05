"""RAG context construction and time-aware train/val/test splitting."""

from __future__ import annotations

from typing import Tuple

import pandas as pd


def add_rag_context(
    events: pd.DataFrame,
    k: int = 3,
    max_chars: int = 4000,
) -> pd.DataFrame:
    """Append a context block of up to ``k`` prior filings for the same ticker.

    Retrieval is purely time-aware: most recent prior events for the same
    ticker, joined with a header. The combined text is truncated to
    ``max_chars`` characters.
    """
    events = events.sort_values("accepted_dt").reset_index(drop=True)
    grouped = events.groupby("ticker")

    rag_texts = []
    for _, row in events.iterrows():
        ticker = row["ticker"]
        accepted = row["accepted_dt"]
        grp = grouped.get_group(ticker)
        prior = (
            grp[grp["accepted_dt"] < accepted]
            .sort_values("accepted_dt", ascending=False)
        )
        context_parts = []
        for _, prow in prior.head(k).iterrows():
            context_parts.append(f"[PAST {prow['filing_date'].date()}]\n{prow['text']}")
        context_str = "\n\n".join(context_parts)

        combined = row["text"]
        if context_str:
            combined = combined + "\n\n[CONTEXT FROM PAST FILINGS]\n" + context_str
        if len(combined) > max_chars:
            combined = combined[:max_chars]
        rag_texts.append(combined)

    events = events.copy()
    events["rag_text"] = rag_texts
    return events


def make_time_splits(
    events: pd.DataFrame,
    train_q: float = 0.6,
    val_q: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-ordered train/val/test split with leakage-reducing buffers.

    * Train: events whose ``label_end_dt`` is at or before the ``train_q``
      quantile of all label-end times.
    * Val: events whose ``accepted_dt`` is at or after the train cutoff and
      whose ``label_end_dt`` is at or before the val cutoff.
    * Test: events whose ``accepted_dt`` is at or after the val cutoff.

    Raises ValueError if any split is empty.
    """
    events = events.sort_values("accepted_dt").reset_index(drop=True)
    t1 = events["label_end_dt"].quantile(train_q)
    t2 = events["label_end_dt"].quantile(val_q)

    train_mask = events["label_end_dt"] <= t1
    val_mask = (events["accepted_dt"] >= t1) & (events["label_end_dt"] <= t2)
    test_mask = events["accepted_dt"] >= t2

    train_df = events[train_mask].reset_index(drop=True)
    val_df = events[val_mask].reset_index(drop=True)
    test_df = events[test_mask].reset_index(drop=True)

    print(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("One of the splits is empty; adjust quantiles or date window.")
    return train_df, val_df, test_df
