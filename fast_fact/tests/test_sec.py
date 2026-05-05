"""Tests for fast_fact.data.sec — all network calls are mocked."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from fast_fact.config import DataConfig, SECConfig
from fast_fact.data import sec as sec_mod


class FakeResponse:
    def __init__(self, *, status_code=200, json_data=None, text="", content=None,
                 content_type="text/html"):
        self.status_code = status_code
        self._json = json_data
        self.text = text
        self.content = content if content is not None else text.encode()
        self.headers = {"Content-Type": content_type}

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_sec_headers_includes_user_agent():
    cfg = SECConfig(user_agent="Foo Bar foo@bar.com")
    headers = sec_mod._sec_headers(cfg)
    assert headers["User-Agent"] == "Foo Bar foo@bar.com"
    assert "Accept-Encoding" in headers


def test_load_ticker_to_cik_map():
    cfg = SECConfig()
    payload = {
        "0": {"ticker": "aapl", "cik_str": 320193},
        "1": {"ticker": "MSFT", "cik_str": 789019},
    }
    with patch.object(sec_mod.requests, "get", return_value=FakeResponse(json_data=payload)):
        m = sec_mod.load_ticker_to_cik_map(cfg)
    assert m["AAPL"] == "0000320193"
    assert m["MSFT"] == "0000789019"


def test_fetch_company_submissions():
    cfg = SECConfig()
    with patch.object(sec_mod.requests, "get",
                      return_value=FakeResponse(json_data={"filings": {"recent": {}}})):
        out = sec_mod.fetch_company_submissions("0000320193", cfg)
    assert out == {"filings": {"recent": {}}}


def _html_with_item(item: str = "Item 2.02", filler: str = "x") -> bytes:
    body = (
        "<html><body>"
        f"<p>{item} Results of Operations</p>"
        f"<div>{filler * 600}</div>"
        "<script>bad()</script>"
        "<style>body{}</style>"
        "</body></html>"
    )
    return body.encode()


def test_extract_text_keeps_item_2_02():
    cfg = SECConfig(min_chars=100, item_filter=True)
    resp = FakeResponse(content=_html_with_item("Item 2.02"), content_type="text/html")
    with patch.object(sec_mod.requests, "get", return_value=resp):
        text = sec_mod.extract_text_from_filing("http://x/y", cfg)
    assert text is not None
    assert "item 2.02" in text.lower()


def test_extract_text_drops_when_no_target_item():
    cfg = SECConfig(min_chars=10, item_filter=True)
    resp = FakeResponse(content=_html_with_item("Item 9.99"), content_type="text/html")
    with patch.object(sec_mod.requests, "get", return_value=resp):
        assert sec_mod.extract_text_from_filing("http://x/y", cfg) is None


def test_extract_text_drops_when_too_short():
    cfg = SECConfig(min_chars=10_000, item_filter=False)
    resp = FakeResponse(content=b"<p>tiny</p>", content_type="text/html")
    with patch.object(sec_mod.requests, "get", return_value=resp):
        assert sec_mod.extract_text_from_filing("http://x/y", cfg) is None


def test_extract_text_handles_non_html_content():
    cfg = SECConfig(min_chars=10, item_filter=False)
    body = "Plain text\r\nIs fine here\r\nover several lines"
    resp = FakeResponse(text=body, content_type="text/plain")
    with patch.object(sec_mod.requests, "get", return_value=resp):
        text = sec_mod.extract_text_from_filing("http://x/y", cfg)
    assert "Plain text" in text
    assert "\r" not in text


def test_extract_text_returns_none_on_404():
    cfg = SECConfig()
    resp = FakeResponse(status_code=404, content=b"")
    with patch.object(sec_mod.requests, "get", return_value=resp):
        assert sec_mod.extract_text_from_filing("http://x/y", cfg) is None


def test_extract_text_swallows_exceptions():
    cfg = SECConfig()
    with patch.object(sec_mod.requests, "get", side_effect=RuntimeError("boom")):
        assert sec_mod.extract_text_from_filing("http://x/y", cfg) is None


def _build_submissions_payload(rows):
    """Build a SEC submissions JSON shaped like the real recent-filings block."""
    keys = ["accessionNumber", "form", "filingDate", "acceptanceDateTime", "primaryDocument"]
    out = {k: [r[k] for r in rows] for k in keys}
    return {"filings": {"recent": out}}


def test_collect_8k_events_for_universe(monkeypatch):
    sec_cfg = SECConfig(min_chars=10, item_filter=False, max_filings_per_ticker=10)
    data_cfg = DataConfig(
        tickers=["AAPL", "ZZZZ"],
        start_date="2020-01-01", end_date="2020-12-31",
    )

    monkeypatch.setattr(sec_mod, "load_ticker_to_cik_map", lambda cfg: {"AAPL": "0000000001"})

    submissions = _build_submissions_payload([
        {"accessionNumber": "0000-1", "form": "8-K", "filingDate": "2020-06-01",
         "acceptanceDateTime": "2020-06-01T10:00:00", "primaryDocument": "a.htm"},
        {"accessionNumber": "0000-2", "form": "10-K", "filingDate": "2020-06-15",
         "acceptanceDateTime": "2020-06-15T10:00:00", "primaryDocument": "b.htm"},
        {"accessionNumber": "0000-3", "form": "8-K", "filingDate": "2019-01-01",
         "acceptanceDateTime": "2019-01-01T10:00:00", "primaryDocument": "old.htm"},
    ])
    monkeypatch.setattr(sec_mod, "fetch_company_submissions", lambda cik, cfg: submissions)
    monkeypatch.setattr(sec_mod, "extract_text_from_filing", lambda url, cfg: "filing body text")

    df = sec_mod.collect_8k_events_for_universe(data_cfg, sec_cfg, sleep_seconds=0)
    assert len(df) == 1  # only the 2020-06-01 8-K survives the date+form filter
    row = df.iloc[0]
    assert row["ticker"] == "AAPL"
    assert row["cik"] == "0000000001"
    assert row["filing_url"].endswith("a.htm")


def test_collect_8k_skips_when_no_8k_matches(monkeypatch):
    sec_cfg = SECConfig(min_chars=10, item_filter=False)
    data_cfg = DataConfig(
        tickers=["AAPL"], start_date="2020-01-01", end_date="2020-12-31",
    )
    monkeypatch.setattr(sec_mod, "load_ticker_to_cik_map", lambda cfg: {"AAPL": "0000000001"})
    monkeypatch.setattr(
        sec_mod, "fetch_company_submissions",
        lambda cik, cfg: _build_submissions_payload([
            {"accessionNumber": "x", "form": "10-K", "filingDate": "2020-06-01",
             "acceptanceDateTime": "2020-06-01T10:00:00", "primaryDocument": "a.htm"},
        ]),
    )
    with pytest.raises(ValueError):
        sec_mod.collect_8k_events_for_universe(data_cfg, sec_cfg, sleep_seconds=0)


def test_collect_8k_skips_unknown_ticker_and_errors_when_empty(monkeypatch):
    sec_cfg = SECConfig()
    data_cfg = DataConfig(tickers=["NOPE"], start_date="2020-01-01", end_date="2020-12-31")
    monkeypatch.setattr(sec_mod, "load_ticker_to_cik_map", lambda cfg: {"AAPL": "0000000001"})
    with pytest.raises(ValueError):
        sec_mod.collect_8k_events_for_universe(data_cfg, sec_cfg, sleep_seconds=0)


def test_collect_8k_skips_filing_with_none_text(monkeypatch):
    sec_cfg = SECConfig(min_chars=10, item_filter=False)
    data_cfg = DataConfig(tickers=["AAPL"], start_date="2020-01-01", end_date="2020-12-31")
    monkeypatch.setattr(sec_mod, "load_ticker_to_cik_map", lambda cfg: {"AAPL": "0000000001"})
    monkeypatch.setattr(
        sec_mod, "fetch_company_submissions",
        lambda cik, cfg: _build_submissions_payload([
            {"accessionNumber": "x", "form": "8-K", "filingDate": "2020-06-01",
             "acceptanceDateTime": "2020-06-01T10:00:00", "primaryDocument": "a.htm"},
        ]),
    )
    monkeypatch.setattr(sec_mod, "extract_text_from_filing", lambda *a, **kw: None)
    with pytest.raises(ValueError):
        sec_mod.collect_8k_events_for_universe(data_cfg, sec_cfg, sleep_seconds=0)


def test_collect_8k_handles_no_recent_filings(monkeypatch):
    sec_cfg = SECConfig()
    data_cfg = DataConfig(tickers=["AAPL"], start_date="2020-01-01", end_date="2020-12-31")
    monkeypatch.setattr(sec_mod, "load_ticker_to_cik_map", lambda cfg: {"AAPL": "0000000001"})
    monkeypatch.setattr(sec_mod, "fetch_company_submissions",
                        lambda cik, cfg: {"filings": {"recent": {}}})
    with pytest.raises(ValueError):
        sec_mod.collect_8k_events_for_universe(data_cfg, sec_cfg, sleep_seconds=0)
