# fast_fact

Event-driven stock direction prediction from SEC 8-K filings.

## 1. Purpose

`fast_fact` is a small research package built around a single empirical
question: *given the text of an 8-K filing, can a fine-tuned language model
predict the sign of the stock's market-adjusted return over the following
trading week?*

To answer it, the package collects 8-Ks from SEC EDGAR, computes
horizon-window abnormal returns versus SPY, and trains and compares four
classifiers on the resulting labeled dataset:

1. **Untuned base prior** — a pre-trained encoder used zero-shot.
2. **RAG baseline** — frozen encoder, classifier head trained on the
   filing concatenated with prior filings from the same ticker.
3. **LoRA SFT** — supervised fine-tuning of the encoder via low-rank
   adapters on the filing text alone.
4. **DPO** — a preference-style fine-tune on top of the LoRA SFT model,
   using the realized label as the chosen class and its complement as the
   rejected class.

The pipeline reports AUC, F1, balanced accuracy, and Brier score on a
held-out time slice, plus an equal-weight long/short paper portfolio and a
"conflict slice" — events where the untuned prior is confidently wrong.

## 2. Dataset

Two public sources, joined on ticker × date:

- **SEC EDGAR** — 8-K filings via
  `https://www.sec.gov/files/company_tickers.json` (ticker→CIK) and
  `https://data.sec.gov/submissions/CIK<...>.json` (per-company submission
  index). Filings are downloaded from
  `https://www.sec.gov/Archives/edgar/data/<cik>/<accession>/<doc>`. By
  default the package keeps only 8-Ks containing Item 2.02 (results of
  operations) or Item 4.02 (non-reliance on prior financials).
- **Daily prices** — adjusted closes for each ticker plus SPY via the
  [`yfinance`](https://pypi.org/project/yfinance/) package. SPY is used as
  the market factor in a per-ticker CAPM beta, and the 5-day post-event
  abnormal return drives the binary label.

Default ticker universe: 30 large-cap US equities across tech, financials,
energy, consumer, and industrials (`AAPL`, `MSFT`, `JPM`, `XOM`, …; see
`fast_fact/cli.py`). Default date window: 2011-01-01 through 2024-12-31.

The SEC requires a descriptive User-Agent with contact info on every
request — set `--user-agent "Your Name your@email"` (or pass `user_agent=`
to `SECConfig`) before running anything that hits EDGAR.

## 3. Installation

Clone (or copy) the repository, then from the package root:

```bash
# create and activate a fresh virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# install the package and its dependencies
pip install -e .

# install dev extras (pytest, coverage) if you want to run the tests
pip install -e ".[dev]"
```

The package targets Python ≥ 3.9 and pulls in `torch`, `transformers`,
`peft`, `yfinance`, and the usual scientific stack. GPU support is
optional — the training loops fall back to CPU when CUDA is unavailable.

## 4. Importing and running

### As a library

```python
from fast_fact import DataConfig, ModelConfig, SECConfig
from fast_fact.pipeline import run_experiment

sec_cfg = SECConfig(user_agent="Your Name your@email")
data_cfg = DataConfig(
    tickers=["AAPL", "MSFT", "JPM"],
    start_date="2018-01-01",
    end_date="2023-12-31",
    horizon_days=5,
    label_threshold=0.005,
)
model_cfg = ModelConfig(base_model="roberta-base", num_epochs=3)

summary = run_experiment(data_cfg, sec_cfg, model_cfg, outdir="runs/demo")
```

You can also call the individual stages directly:

```python
from fast_fact.data.sec import collect_8k_events_for_universe
from fast_fact.data.prices import (
    download_price_history, attach_abnormal_returns_and_labels,
)
from fast_fact.data.splits import add_rag_context, make_time_splits

events = collect_8k_events_for_universe(data_cfg, sec_cfg)
prices = download_price_history(data_cfg.tickers, data_cfg.start_date, data_cfg.end_date)
labeled = attach_abnormal_returns_and_labels(events, prices, data_cfg)
ragged = add_rag_context(labeled)
train_df, val_df, test_df = make_time_splits(ragged)
```

### As a CLI

`pip install` registers a `fast-fact` entry point:

```bash
fast-fact \
    --tickers AAPL MSFT JPM \
    --start-date 2018-01-01 --end-date 2023-12-31 \
    --user-agent "Your Name your@email" \
    --outdir runs/demo
```

Equivalent module form: `python -m fast_fact.cli ...`. All flags are
documented under `fast-fact --help`; flags can also be supplied via a JSON
file passed as `--config path/to/config.json`.

### Running the tests

```bash
pytest                                       # run the suite
pytest --cov=fast_fact --cov-report=term-missing   # with coverage
```

Coverage is configured in `pyproject.toml`; the suite mocks all network and
HuggingFace-model calls so it runs in seconds without GPU or internet.

## File layout

```
fast_fact/
├── README.md
├── pyproject.toml
├── requirements.txt
├── fast_fact/
│   ├── __init__.py
│   ├── config.py            # SECConfig, DataConfig, ModelConfig
│   ├── pipeline.py          # prepare_events, run_experiment
│   ├── cli.py               # `fast-fact` entry point
│   ├── data/
│   │   ├── sec.py           # SEC EDGAR fetching
│   │   ├── prices.py        # yfinance + abnormal returns
│   │   ├── splits.py        # RAG context + time splits
│   │   └── datasets.py      # PyTorch Dataset wrappers
│   └── models/
│       ├── base.py          # build_base_model, apply_lora, freeze_encoder
│       ├── train.py         # train_supervised_lora, train_rag_baseline, train_dpo
│       └── evaluate.py      # metrics, FinBERT prior, paper portfolio
└── tests/                   # pytest suite (>80% line coverage)
```

## 5. AI Usage

The research direction and experimental design behind this project are
ours. Alex and I arrived at the 8-K event-driven formulation by iterating
through several alternatives including pure price-momentum baselines, sentiment-only
news classifiers, and longer-horizon 10-K comparisons. Our work centered on design and decisoin making like 
choosing the ticker universe, the 2011–2024 date window, the Item 2.02 / 4.02 filter for filings that actually carry
new information, the beta-adjusted abnormal-return construction against
SPY, the time-split boundaries chosen to avoid look-ahead leakage, and the
four-way model comparison (untuned prior, frozen-encoder RAG, LoRA SFT,
DPO) designed so each contrast isolates a specific source of signal we
hypothesized would matter. We used ChatGPT and Claude Code as
implementation aids and we gave them detailed instructions on how to impelment
individual or groups of functions. We also had to explain unfamiliar APIs (PEFT, the DPO
objective, yfinance quirks), and accelerate boilerplate like the training
loops and unit-test fixtures. Their output was reviewed, edited, and
integrated by us rather and the conceptual work,
the data-handling decisions, and the verification of correctness against
our intent remained our responsibility throughout.
