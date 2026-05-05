from fast_fact.config import DataConfig, ModelConfig, SECConfig


def test_sec_config_defaults():
    cfg = SECConfig()
    assert "@" in cfg.user_agent  # contact info expected
    assert cfg.min_chars == 500
    assert cfg.item_filter is True
    assert cfg.max_filings_per_ticker == 200


def test_sec_config_overrides():
    cfg = SECConfig(user_agent="A B a@b.com", min_chars=10, item_filter=False, max_filings_per_ticker=5)
    assert cfg.min_chars == 10
    assert cfg.item_filter is False
    assert cfg.max_filings_per_ticker == 5


def test_data_config_defaults_and_override():
    cfg = DataConfig()
    assert cfg.tickers == []
    assert cfg.horizon_days == 5
    assert cfg.label_threshold == 0.01

    cfg2 = DataConfig(tickers=["AAPL"], horizon_days=3, label_threshold=0.02)
    assert cfg2.tickers == ["AAPL"]
    assert cfg2.horizon_days == 3
    assert cfg2.label_threshold == 0.02


def test_model_config_defaults():
    cfg = ModelConfig()
    assert cfg.base_model == "roberta-base"
    assert cfg.lora_r == 16
    assert 0.0 < cfg.dpo_beta <= 1.0


def test_data_configs_are_independent():
    a = DataConfig()
    b = DataConfig()
    a.tickers.append("AAPL")
    assert b.tickers == []  # default_factory should not share state
