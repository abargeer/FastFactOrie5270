"""CLI plumbing tests — run_experiment is stubbed to keep things fast."""

from __future__ import annotations

import json

import pytest

from fast_fact import cli as cli_mod
from fast_fact.config import DataConfig, ModelConfig, SECConfig


def test_build_parser_has_core_flags():
    p = cli_mod.build_parser()
    args = p.parse_args(["--tickers", "AAPL", "MSFT", "--num-epochs", "2"])
    assert args.tickers == ["AAPL", "MSFT"]
    assert args.num_epochs == 2


def test_main_dispatches_to_run_experiment(monkeypatch, tmp_path):
    captured = {}

    def stub_run(data_cfg, sec_cfg, model_cfg, outdir, use_cache):
        captured["data_cfg"] = data_cfg
        captured["sec_cfg"] = sec_cfg
        captured["model_cfg"] = model_cfg
        captured["outdir"] = outdir
        captured["use_cache"] = use_cache
        return {"ok": True}

    monkeypatch.setattr(cli_mod, "run_experiment", stub_run)
    out = cli_mod.main([
        "--tickers", "AAPL",
        "--start-date", "2020-01-01", "--end-date", "2020-06-30",
        "--user-agent", "T E test@example.com",
        "--max-filings-per-ticker", "5",
        "--base-model", "tiny", "--max-length", "16",
        "--batch-size", "1", "--num-epochs", "1",
        "--outdir", str(tmp_path),
        "--no-cache",
    ])
    assert out == {"ok": True}
    assert captured["data_cfg"].tickers == ["AAPL"]
    assert captured["sec_cfg"].user_agent == "T E test@example.com"
    assert captured["sec_cfg"].max_filings_per_ticker == 5
    assert captured["model_cfg"].base_model == "tiny"
    assert captured["use_cache"] is False


def test_apply_config_file_overrides_defaults(monkeypatch, tmp_path):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"num_epochs": 7, "batch_size": 32}))

    monkeypatch.setattr(cli_mod, "run_experiment",
                        lambda *a, **kw: {"num_epochs": kw or a})
    captured = {}

    def stub_run(data_cfg, sec_cfg, model_cfg, outdir, use_cache):
        captured["model_cfg"] = model_cfg
        return {}

    monkeypatch.setattr(cli_mod, "run_experiment", stub_run)
    cli_mod.main(["--config", str(cfg_path)])
    assert captured["model_cfg"].num_epochs == 7
    assert captured["model_cfg"].batch_size == 32


def test_apply_config_file_does_not_override_explicit_cli(monkeypatch, tmp_path):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"num_epochs": 7}))

    captured = {}

    def stub_run(data_cfg, sec_cfg, model_cfg, outdir, use_cache):
        captured["model_cfg"] = model_cfg

    monkeypatch.setattr(cli_mod, "run_experiment", stub_run)
    cli_mod.main(["--num-epochs", "3", "--config", str(cfg_path)])
    assert captured["model_cfg"].num_epochs == 3  # explicit CLI wins
