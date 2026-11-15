# AI Pullback Prediction

Research-grade, probabilistic XAUUSD pullback prediction system. The initial target is **XAUUSD on M1 and M5**, with a chronological, leakage-safe data pipeline and a decision interface that emits **BUY / SELL / WAIT**. It does not place broker orders.

## Project principles

- Use Dukascopy bid and ask history as the canonical market-data source.
- Train only on 2015-01-01 through 2020-12-31.
- Validate on 2021-01-01 through 2023-12-31.
- Keep 2024-01-01 onward as a final, untouched out-of-sample test period.
- Preserve bid/ask prices; model spread explicitly rather than treating mid-price as executable.
- Use walk-forward evaluation, realistic spread/slippage assumptions, and uncertainty-aware abstention.
- Keep live order execution disabled. Predictions are research signals, not financial advice.

## Current implementation

This repository is being built in verifiable stages. The first stage establishes the data contract, chronological splits, feature engineering, event labels, and tests. Model training, ensemble calibration, backtesting, API, dashboard, and deployment are subsequent integration stages.

### Data contract

Input CSV files must contain timestamp and bid/ask OHLC prices. Tick volume is optional. Timestamps are parsed as UTC and normalized to a consistent candle interval. Raw files are never overwritten.

Accepted column aliases are documented in `src/pullback_prediction/data/loader.py`.

## Quick start

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e ".[dev]"
pytest
```

Validate a downloaded Dukascopy CSV:

```bash
pullback-predict validate-data --input data/raw/XAUUSD_M1.csv --timeframe M1
```

Create leakage-safe date partitions:

```bash
pullback-predict split-data --input data/processed/XAUUSD_M1.parquet --output-dir data/splits
```

## Planned architecture

1. **Data:** Dukascopy bid/ask ingestion, integrity checks, resampling, spread and session metadata.
2. **Features:** trend, volatility, momentum, candle geometry, market structure and liquidity proxies, calculated causally.
3. **Labels:** future-pathrier event labels with configurable horizon and barriers; no future data enters features.
4. **Models:** gradient-boosted tabular model, sequence LSTM and Transformer, regime detector, calibrated meta-learner.
5. **Decision:** BUY / SELL / WAIT with calibrated probabilities, expected value, uncertainty and risk levels.
6. **Evaluation:** chronological holdout, walk-forward tests, spread/slippage/commission-aware backtests and benchmark comparisons.
7. **Serving:** FastAPI prediction service and Streamlit research dashboard.
8. **Operations:** Docker/VPS deployment, reproducible configs, monitoring and paper-trading only.

## Safety boundary

There is no broker execution module. No credentials are requested or stored. Any future live-trading integration must remain out of scope unless explicitly redesigned and separately authorized.

## Status

See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for verified, pending, and blocked work. A component is not marked verified until its checks have actually been run.
