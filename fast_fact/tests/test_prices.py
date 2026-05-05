"""Tests for fast_fact.data.prices — yfinance is stubbed via the yf_module hook."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fast_fact.config import DataConfig
from fast_fact.data import prices as prices_mod


def _make_price_df(start, periods, close=100.0, drift=0.001, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=periods)
    rets = rng.normal(drift, 0.01, size=periods)
    closes = close * np.exp(np.cumsum(rets))
    return pd.DataFrame({"Close": closes}, index=idx)


def test_compute_beta_basic():
    rng = np.random.default_rng(1)
    n = 200
    rm = pd.Series(rng.normal(0, 0.01, size=n))
    rs = 1.5 * rm + pd.Series(rng.normal(0, 0.005, size=n))
    beta = prices_mod.compute_beta(rs, rm)
    assert beta == pytest.approx(1.5, abs=0.2)


def test_compute_beta_too_few_obs_returns_none():
    a = pd.Series(np.zeros(5))
    b = pd.Series(np.zeros(5))
    assert prices_mod.compute_beta(a, b) is None


def test_compute_beta_zero_market_variance_returns_none():
    n = 100
    a = pd.Series(np.random.default_rng(0).normal(size=n))
    b = pd.Series(np.zeros(n))
    assert prices_mod.compute_beta(a, b) is None


def test_download_price_history_uses_injected_yf_module():
    """yf_module=stub lets us avoid real yfinance + network."""
    aapl = _make_price_df("2020-01-01", 50)
    spy = _make_price_df("2020-01-01", 50, drift=0.0005, seed=2)
    multi = pd.concat({"AAPL": aapl, "SPY": spy}, axis=1)

    class StubYF:
        def download(self, **kwargs):
            return multi

    out = prices_mod.download_price_history(
        ["AAPL"], "2020-01-01", "2020-03-01", yf_module=StubYF()
    )
    assert "AAPL" in out and "SPY" in out
    assert "Close" in out["AAPL"].columns


def test_download_price_history_warns_for_missing_ticker(capsys):
    aapl = _make_price_df("2020-01-01", 30)
    multi = pd.concat({"AAPL": aapl}, axis=1)

    class StubYF:
        def download(self, **kwargs):
            return multi

    out = prices_mod.download_price_history(
        ["AAPL"], "2020-01-01", "2020-02-01", yf_module=StubYF()
    )
    assert "SPY" not in out
    captured = capsys.readouterr().out
    assert "No data for ticker SPY" in captured


def test_download_price_history_handles_single_index():
    aapl = _make_price_df("2020-01-01", 30)

    class StubYF:
        def download(self, **kwargs):
            return aapl

    out = prices_mod.download_price_history(
        ["AAPL"], "2020-01-01", "2020-02-01", yf_module=StubYF()
    )
    assert len(out) == 1


def test_attach_abnormal_returns_and_labels_assigns_labels():
    """End-to-end labeling on a synthetic universe with strong AAA returns."""
    aaa_idx = pd.bdate_range("2020-01-01", periods=200)
    spy_idx = pd.bdate_range("2020-01-01", periods=200)

    rng = np.random.default_rng(7)
    aaa_close = 100 * np.exp(np.cumsum(rng.normal(0.005, 0.005, size=200)))  # strong drift
    spy_close = 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.005, size=200)))
    aaa = pd.DataFrame({"Close": aaa_close}, index=aaa_idx)
    spy = pd.DataFrame({"Close": spy_close}, index=spy_idx)
    prices = {"AAA": aaa, "SPY": spy}

    accepted = pd.Timestamp("2020-03-15 10:00:00", tz="US/Eastern")
    events = pd.DataFrame({
        "ticker": ["AAA"],
        "accepted_dt": [accepted],
        "filing_date": [accepted.date()],
        "text": ["x"],
    })
    cfg = DataConfig(tickers=["AAA"], horizon_days=5, label_threshold=0.001)
    out = prices_mod.attach_abnormal_returns_and_labels(events, prices, cfg)
    assert len(out) == 1
    assert out.iloc[0]["label"] in (0, 1)
    assert pd.notna(out.iloc[0]["abn_ret"])


def test_attach_handles_after_hours_entry_and_no_data_ticker():
    aaa = pd.DataFrame(
        {"Close": np.linspace(100, 110, 200)},
        index=pd.bdate_range("2020-01-01", periods=200),
    )
    spy = pd.DataFrame(
        {"Close": np.linspace(100, 100, 200)},
        index=pd.bdate_range("2020-01-01", periods=200),
    )
    prices = {"AAA": aaa, "SPY": spy}

    after_hours = pd.Timestamp("2020-03-13 18:00:00", tz="US/Eastern")
    far_future = pd.Timestamp("2099-01-01 10:00:00", tz="US/Eastern")
    events = pd.DataFrame({
        "ticker": ["AAA", "AAA", "ZZZ"],
        "accepted_dt": [after_hours, far_future, after_hours],
        "filing_date": [after_hours.date(), far_future.date(), after_hours.date()],
        "text": ["x"] * 3,
    })
    cfg = DataConfig(tickers=["AAA"], horizon_days=5, label_threshold=0.001)
    out = prices_mod.attach_abnormal_returns_and_labels(events, prices, cfg)
    assert len(out) >= 1
    assert (out["ticker"] == "AAA").all()


def test_attach_drops_when_label_is_ambiguous():
    flat_idx = pd.bdate_range("2020-01-01", periods=200)
    flat = pd.DataFrame({"Close": np.full(200, 100.0)}, index=flat_idx)
    prices = {"AAA": flat, "SPY": flat.copy()}
    accepted = pd.Timestamp("2020-03-15 10:00:00", tz="US/Eastern")
    events = pd.DataFrame({
        "ticker": ["AAA"], "accepted_dt": [accepted],
        "filing_date": [accepted.date()], "text": ["x"],
    })
    cfg = DataConfig(tickers=["AAA"], horizon_days=5, label_threshold=0.5)
    out = prices_mod.attach_abnormal_returns_and_labels(events, prices, cfg)
    assert len(out) == 0  # zero abnormal return below 0.5 threshold => dropped


def test_attach_raises_without_spy():
    flat = pd.DataFrame(
        {"Close": np.full(200, 100.0)},
        index=pd.bdate_range("2020-01-01", periods=200),
    )
    prices = {"AAA": flat}
    events = pd.DataFrame({
        "ticker": ["AAA"], "accepted_dt": [pd.Timestamp("2020-03-15", tz="US/Eastern")],
        "filing_date": ["2020-03-15"], "text": ["x"],
    })
    cfg = DataConfig(tickers=["AAA"])
    with pytest.raises(ValueError):
        prices_mod.attach_abnormal_returns_and_labels(events, prices, cfg)


def test_attach_warns_when_close_missing(capsys):
    idx = pd.bdate_range("2020-01-01", periods=200)
    spy = pd.DataFrame({"Close": np.linspace(100, 101, 200)}, index=idx)
    aaa_no_close = pd.DataFrame({"Open": np.linspace(100, 101, 200)}, index=idx)
    prices = {"SPY": spy, "AAA": aaa_no_close}
    accepted = pd.Timestamp("2020-03-15 10:00:00", tz="US/Eastern")
    events = pd.DataFrame({
        "ticker": ["AAA"], "accepted_dt": [accepted],
        "filing_date": [accepted.date()], "text": ["x"],
    })
    cfg = DataConfig(tickers=["AAA"], horizon_days=5, label_threshold=0.001)
    out = prices_mod.attach_abnormal_returns_and_labels(events, prices, cfg)
    assert len(out) == 0  # AAA dropped, no labels
    assert "No 'Close'" in capsys.readouterr().out
