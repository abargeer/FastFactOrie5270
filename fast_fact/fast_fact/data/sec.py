"""Fetching 8-K filings from SEC EDGAR.

The SEC requires a descriptive User-Agent with contact info on every request;
the rest of these helpers are thin wrappers around the public JSON / archive
endpoints.
"""

from __future__ import annotations

import time
import uuid
from typing import Dict, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from fast_fact.config import DataConfig, SECConfig

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_BASE_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"


def _sec_headers(sec_cfg: SECConfig) -> Dict[str, str]:
    """Return the minimum SEC-compliant request headers.

    The Host header is intentionally not set so requests can derive it from
    each URL (data.sec.gov and www.sec.gov both serve content here).
    """
    return {
        "User-Agent": sec_cfg.user_agent,
        "Accept-Encoding": "gzip, deflate",
    }


def load_ticker_to_cik_map(sec_cfg: SECConfig) -> Dict[str, str]:
    """Load the official ticker->CIK mapping from SEC.

    Returns:
        Dict from upper-cased ticker symbol to 10-digit zero-padded CIK string.
    """
    resp = requests.get(SEC_TICKERS_URL, headers=_sec_headers(sec_cfg))
    resp.raise_for_status()
    data = resp.json()
    mapping: Dict[str, str] = {}
    for _, row in data.items():
        ticker = row["ticker"].upper()
        cik_str = str(row["cik_str"]).zfill(10)
        mapping[ticker] = cik_str
    return mapping


def fetch_company_submissions(cik: str, sec_cfg: SECConfig) -> Dict:
    """Fetch the JSON submissions blob for a CIK."""
    url = SEC_SUBMISSIONS_URL.format(cik=cik)
    resp = requests.get(url, headers=_sec_headers(sec_cfg))
    resp.raise_for_status()
    return resp.json()


def extract_text_from_filing(url: str, sec_cfg: SECConfig) -> Optional[str]:
    """Download a filing's primary document and reduce it to plain text.

    Returns None if the request fails, the text is shorter than
    ``sec_cfg.min_chars``, or (when ``sec_cfg.item_filter`` is True) the
    document does not mention Item 2.02 or Item 4.02.
    """
    try:
        resp = requests.get(url, headers=_sec_headers(sec_cfg), timeout=30)
        if resp.status_code != 200:
            return None
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" in content_type.lower():
            soup = BeautifulSoup(resp.content, "lxml")
            for tag in soup(["script", "style"]):
                tag.decompose()
            texts = [
                el.get_text(separator=" ", strip=True)
                for el in soup.find_all(["p", "div"])
            ]
            text = "\n".join(t for t in texts if t)
        else:
            text = resp.text

        text = text.replace("\r", "\n")
        text = "\n".join(line.strip() for line in text.split("\n") if line.strip())

        if len(text) < sec_cfg.min_chars:
            return None

        if sec_cfg.item_filter:
            lower = text.lower()
            if "item 2.02" not in lower and "item 4.02" not in lower:
                return None

        return text
    except Exception as e:  # network/parse errors are non-fatal per filing
        print(f"[WARN] Failed to fetch filing {url}: {e}")
        return None


def collect_8k_events_for_universe(
    data_cfg: DataConfig,
    sec_cfg: SECConfig,
    sleep_seconds: float = 0.2,
) -> pd.DataFrame:
    """Build a DataFrame of 8-K events for the configured ticker universe.

    Columns: ``event_id, ticker, cik, form, filing_date, accepted_dt,
    primary_doc, filing_url, text``.

    The ``sleep_seconds`` argument is exposed mainly so tests can avoid real
    sleeps; the SEC asks for at most ~10 requests per second.
    """
    ticker_to_cik = load_ticker_to_cik_map(sec_cfg)
    print(f"Loaded {len(ticker_to_cik)} ticker->CIK mappings from SEC.")

    rows = []
    start = pd.to_datetime(data_cfg.start_date)
    end = pd.to_datetime(data_cfg.end_date)

    for ticker in data_cfg.tickers:
        up_ticker = ticker.upper()
        if up_ticker not in ticker_to_cik:
            print(f"[WARN] Ticker {ticker} not found in SEC mapping; skipping.")
            continue

        cik = ticker_to_cik[up_ticker]
        print(f"Fetching submissions for {ticker} (CIK {cik})...")
        subs = fetch_company_submissions(cik, sec_cfg)
        filings = subs.get("filings", {}).get("recent", {})
        if not filings:
            print(f"[WARN] No recent filings for {ticker}")
            continue

        df = pd.DataFrame(filings)
        df["filingDate"] = pd.to_datetime(df["filingDate"])
        df["acceptedDateTime"] = pd.to_datetime(df["acceptanceDateTime"])

        mask = (
            (df["form"] == "8-K")
            & (df["filingDate"] >= start)
            & (df["filingDate"] <= end)
        )
        df = df.loc[mask].copy()
        if df.empty:
            print(f"[INFO] No 8-Ks for {ticker} in date window.")
            continue

        df = df.sort_values("acceptedDateTime")
        if sec_cfg.max_filings_per_ticker is not None:
            df = df.head(sec_cfg.max_filings_per_ticker)

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{ticker} 8-Ks"):
            accession = row["accessionNumber"]
            primary_doc = row["primaryDocument"]
            cik_nolead = str(int(cik))
            accession_nodash = accession.replace("-", "")
            filing_url = (
                f"{SEC_BASE_ARCHIVES}/{cik_nolead}/{accession_nodash}/{primary_doc}"
            )

            text = extract_text_from_filing(filing_url, sec_cfg)
            if sleep_seconds:
                time.sleep(sleep_seconds)

            if text is None:
                continue

            rows.append(
                {
                    "event_id": str(uuid.uuid4()),
                    "ticker": up_ticker,
                    "cik": cik,
                    "form": row["form"],
                    "filing_date": row["filingDate"],
                    "accepted_dt": row["acceptedDateTime"],
                    "primary_doc": primary_doc,
                    "filing_url": filing_url,
                    "text": text,
                }
            )

    events_df = pd.DataFrame(rows)
    if events_df.empty:
        raise ValueError("No 8-K events collected. Adjust tickers/date window.")
    events_df = events_df.sort_values("accepted_dt").reset_index(drop=True)
    print(f"Collected {len(events_df)} 8-K events across universe.")
    return events_df
