"""Price downloads, beta estimation, and abnormal-return labeling."""

from __future__ import annotations

import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from fast_fact.config import DataConfig


def download_price_history(
    tickers: List[str],
    start_date: str,
    end_date: str,
    yf_module=None,
) -> Dict[str, pd.DataFrame]:
    """Download daily price history for tickers and SPY via yfinance.

    The ``yf_module`` argument exists so tests can inject a stub. In normal
    use it stays None and yfinance is imported lazily.
    """
    if yf_module is None:  # lazy import keeps the test suite fast
        import yfinance as yf

        yf_module = yf

    all_tickers = sorted(set(tickers + ["SPY"]))
    start = pd.to_datetime(start_date) - pd.Timedelta(days=400)
    end = pd.to_datetime(end_date) + pd.Timedelta(days=10)
    print(f"Downloading daily prices from {start.date()} to {end.date()}...")

    data = yf_module.download(
        tickers=all_tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    prices: Dict[str, pd.DataFrame] = {}
    if isinstance(data.columns, pd.MultiIndex):
        for t in all_tickers:
            if t not in data.columns.get_level_values(0):
                print(f"[WARN] No data for ticker {t}")
                continue
            df_t = data[t].copy()
            df_t.index.name = "date"
            prices[t] = df_t
    else:
        data.index.name = "date"
        prices[all_tickers[0]] = data

    return prices


def compute_beta(
    stock_returns: pd.Series, market_returns: pd.Series
) -> Optional[float]:
    """OLS beta of ``stock_returns`` on ``market_returns``.

    Returns None if fewer than 60 overlapping observations exist or if the
    market return has zero variance over that window.
    """
    df = pd.concat([stock_returns, market_returns], axis=1, join="inner").dropna()
    if len(df) < 60:
        return None
    r_s = df.iloc[:, 0].values
    r_m = df.iloc[:, 1].values
    var_m = np.var(r_m)
    if var_m == 0:
        return None
    cov_sm = np.cov(r_s, r_m)[0, 1]
    return float(cov_sm / var_m)


def attach_abnormal_returns_and_labels(
    events: pd.DataFrame,
    prices: Dict[str, pd.DataFrame],
    data_cfg: DataConfig,
) -> pd.DataFrame:
    """Compute horizon-window abnormal returns vs SPY and assign labels.

    For each event:
      * Convert ``accepted_dt`` to US/Eastern, then drop tz so it aligns with
        yfinance's tz-naive daily index.
      * Choose the entry day: same trading day if accepted before 16:00 ET,
        otherwise the next trading day.
      * Sum log returns over ``data_cfg.horizon_days`` trading days, subtract
        ``beta * SPY return`` over the same window.
      * Label 1 if abn_ret >= ``label_threshold``, 0 if <= -``label_threshold``,
        otherwise drop the event as ambiguous.
    """
    horizon = data_cfg.horizon_days
    thr = data_cfg.label_threshold

    events = events.copy()
    events["accepted_dt"] = pd.to_datetime(events["accepted_dt"], utc=True)
    events["accepted_dt"] = (
        events["accepted_dt"]
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
    )

    returns: Dict[str, pd.DataFrame] = {}
    for ticker, df in prices.items():
        if "Close" not in df.columns:
            print(f"[WARN] No 'Close' in price data for {ticker}; skipping.")
            continue
        df = df.sort_index()
        df["log_ret"] = np.log(df["Close"]).diff()
        returns[ticker] = df

    if "SPY" not in returns:
        raise ValueError("SPY price data missing; cannot compute abnormal returns.")
    mkt = returns["SPY"]

    betas: Dict[str, float] = {}
    for ticker in events["ticker"].unique():
        if ticker not in returns:
            print(f"[WARN] No returns for ticker {ticker}; skipping beta.")
            continue
        beta = compute_beta(returns[ticker]["log_ret"], mkt["log_ret"])
        if beta is None:
            print(f"[WARN] Could not compute beta for {ticker}; defaulting to 1.0")
            beta = 1.0
        betas[ticker] = beta

    entry_dates = []
    exit_dates = []
    ar_list = []
    labels = []
    label_end_times = []

    def _miss():
        entry_dates.append(None)
        exit_dates.append(None)
        ar_list.append(np.nan)
        labels.append(None)
        label_end_times.append(pd.NaT)

    for _, row in tqdm(events.iterrows(), total=len(events), desc="Labeling events"):
        ticker = row["ticker"]
        if ticker not in returns:
            _miss()
            continue

        df_s = returns[ticker]
        df_m = mkt
        accepted = row["accepted_dt"]

        if accepted.time() >= datetime.time(16, 0):
            entry_date = accepted.normalize() + pd.Timedelta(days=1)
        else:
            entry_date = accepted.normalize()

        while entry_date not in df_s.index:
            entry_date += pd.Timedelta(days=1)
            if entry_date > df_s.index[-1]:
                entry_date = None
                break

        if entry_date is None:
            _miss()
            continue

        try:
            entry_idx = df_s.index.get_loc(entry_date)
        except KeyError:
            _miss()
            continue

        exit_idx = entry_idx + horizon
        if exit_idx >= len(df_s):
            _miss()
            continue

        exit_date = df_s.index[exit_idx]

        rs_slice = df_s.loc[entry_date:exit_date]["log_ret"].dropna()
        rm_slice = df_m.loc[entry_date:exit_date]["log_ret"].dropna()
        r_s_total = rs_slice.sum()
        r_m_total = rm_slice.sum() if len(rm_slice) > 0 else 0.0

        beta = betas.get(ticker, 1.0)
        abn_ret = r_s_total - beta * r_m_total

        if abn_ret >= thr:
            label: Optional[int] = 1
        elif abn_ret <= -thr:
            label = 0
        else:
            label = None

        entry_dates.append(entry_date)
        exit_dates.append(exit_date)
        ar_list.append(abn_ret)
        labels.append(label)
        label_end_times.append(exit_date)

    events["entry_date"] = entry_dates
    events["exit_date"] = exit_dates
    events["abn_ret"] = ar_list
    events["label"] = labels
    events["label_end_dt"] = label_end_times

    events = events.dropna(subset=["entry_date", "exit_date", "abn_ret"])
    events = events[events["label"].notnull()].reset_index(drop=True)
    print(f"Events with usable labels: {len(events)}")
    return events
