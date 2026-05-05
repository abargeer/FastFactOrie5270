import pandas as pd
import pytest

from fast_fact.data.splits import add_rag_context, make_time_splits


def _events(n=12, ticker_a="AAA", ticker_b="BBB"):
    accepted = pd.date_range("2020-01-01", periods=n, freq="7D")
    half = n // 2
    return pd.DataFrame({
        "ticker": [ticker_a] * half + [ticker_b] * (n - half),
        "accepted_dt": accepted,
        "filing_date": accepted,
        "text": [f"body number {i} item 2.02" for i in range(n)],
        "label": [i % 2 for i in range(n)],
        "label_end_dt": accepted + pd.Timedelta(days=5),
    })


def test_add_rag_context_appends_prior_filings():
    df = _events(n=8)
    out = add_rag_context(df, k=2, max_chars=10_000)
    assert "rag_text" in out.columns
    last_aaa = out[out["ticker"] == "AAA"].iloc[-1]
    assert "[CONTEXT FROM PAST FILINGS]" in last_aaa["rag_text"]
    first_aaa = out[out["ticker"] == "AAA"].iloc[0]
    assert "[CONTEXT FROM PAST FILINGS]" not in first_aaa["rag_text"]


def test_add_rag_context_truncates_to_max_chars():
    df = _events(n=4)
    df["text"] = "x" * 5000
    out = add_rag_context(df, k=2, max_chars=200)
    assert all(len(t) <= 200 for t in out["rag_text"])


def test_make_time_splits_three_nonempty():
    df = _events(n=30)
    train, val, test = make_time_splits(df, train_q=0.5, val_q=0.8)
    assert len(train) > 0 and len(val) > 0 and len(test) > 0
    # No row simultaneously in train and test
    assert not set(train["accepted_dt"]).intersection(test["accepted_dt"])


def test_make_time_splits_raises_on_empty_split():
    df = _events(n=4)
    with pytest.raises(ValueError):
        make_time_splits(df, train_q=0.99, val_q=0.999)
