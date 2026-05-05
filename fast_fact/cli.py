"""Command-line entry point for the experiment pipeline.

Configurable via flags or a JSON config file. Example:

    python -m fast_fact.cli --tickers AAPL MSFT --outdir runs/demo
"""

from __future__ import annotations

import argparse
import json
from typing import List, Optional, Sequence

from fast_fact.config import DataConfig, ModelConfig, SECConfig
from fast_fact.pipeline import run_experiment

DEFAULT_TICKERS: List[str] = [
    "AAPL", "MSFT", "AMZN", "META", "GOOG", "NVDA", "TSLA", "NFLX", "AMD", "INTC",
    "JPM", "BAC", "WFC", "C", "GS", "MS",
    "XOM", "CVX", "COP", "OXY",
    "WMT", "TGT", "HD", "LOW", "MCD", "SBUX",
    "T", "VZ", "IBM", "GE",
]


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    p = argparse.ArgumentParser(
        prog="fast_fact",
        description="Train and evaluate 8-K event-driven stock direction models.",
    )
    p.add_argument(
        "--tickers", nargs="+", default=DEFAULT_TICKERS,
        help="Equity tickers to fetch 8-Ks for.",
    )
    p.add_argument("--start-date", default="2011-01-01")
    p.add_argument("--end-date", default="2024-12-31")
    p.add_argument("--horizon-days", type=int, default=5)
    p.add_argument("--label-threshold", type=float, default=0.005)
    p.add_argument(
        "--user-agent", default="fast_fact contact@example.com",
        help="SEC-required descriptive User-Agent with contact info.",
    )
    p.add_argument("--max-filings-per-ticker", type=int, default=300)
    p.add_argument("--base-model", default="roberta-base")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-epochs", type=int, default=5)
    p.add_argument("--outdir", default="fast_fact_outputs")
    p.add_argument(
        "--config", default=None,
        help="Path to a JSON config file. CLI flags override file values.",
    )
    p.add_argument(
        "--no-cache", action="store_true",
        help="Disable parquet caching of intermediate event files.",
    )
    return p


def _apply_config_file(args: argparse.Namespace) -> argparse.Namespace:
    """Merge values from --config into args (CLI flags win on conflict)."""
    if not args.config:
        return args
    with open(args.config, "r") as f:
        cfg = json.load(f)
    parser = build_parser()
    defaults = {a.dest: a.default for a in parser._actions}
    for k, v in cfg.items():
        # only override defaults — explicit CLI flags keep their value
        if hasattr(args, k) and getattr(args, k) == defaults.get(k):
            setattr(args, k, v)
    return args


def main(argv: Optional[Sequence[str]] = None) -> dict:
    """Parse CLI args and dispatch to :func:`fast_fact.pipeline.run_experiment`."""
    args = build_parser().parse_args(argv)
    args = _apply_config_file(args)

    sec_cfg = SECConfig(
        user_agent=args.user_agent,
        max_filings_per_ticker=args.max_filings_per_ticker,
    )
    data_cfg = DataConfig(
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        horizon_days=args.horizon_days,
        label_threshold=args.label_threshold,
    )
    model_cfg = ModelConfig(
        base_model=args.base_model,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
    )
    return run_experiment(
        data_cfg, sec_cfg, model_cfg,
        outdir=args.outdir, use_cache=not args.no_cache,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
