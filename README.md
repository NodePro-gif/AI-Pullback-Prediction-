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

The first stage covers data contracts, chronological splits, causal features, barrier labels, and automated tests. A first HistGradientBoosting baseline is available for train/validation experiments. It fits only on training rows and creates labels separately within each partition so target horizons cannot cross split boundaries.

### Data contract

Input CSV files must contain timestamp and bid/ask OHLC prices. Tick volume is optional. Timestamps are parsed as UTC and normalized to a consistent candle interval. Raw files are never overwritten. Combined bid/ask files and separate bid and ask exports are supported.

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

Create chronological partitions:

```bash
pullback-predict split-data --input data/processed/XAUUSD_M1.parquet --output-dir data/splits
```

Train the baseline on 2015–2020 and evaluate on 2021–2023:

```bash
pullback-predict train-baseline --input data/processed/XAUUSD_M1.parquet --output artifacts/xauusd_m1_baseline.joblib
```

The command writes a model artifact and sibling `.metrics.json`. The 2024+ final-test partition is excluded. Run separately for M5 data.

Generate out-of-sample or validation predictions from a saved model. The model emits a signal only when its highest raw class score meets the abstention threshold; these scores are not calibrated probabilities:

```bash
pullback-predict predict --input data/processed/XAUUSD_M1.parquet --model artifacts/xauusd_baseline.joblib --output data/processed/XAUUSD_M1_signals.parquet --min-probability 0.60
```

Then backtest the resulting parquet, which contains `signal` (-1 short, 0 wait, +1 long) and `atr_14` alongside bid/ask OHLC:

```bash
pullback-predict backtest-signals --input data/processed/XAUUSD_M1_signals.parquet --output-dir artifacts/backtest_m1 --slippage-price 0.05
```

The simulator enters at the next candle open, uses ask for long entries and bid for short entries, and exits against the opposite quote. Stop/target collisions within one candle are resolved conservatively as stop-loss first. PnL is price units per unit, not account currency or lot sizing.

## Planned architecture

1. **Data:** Dukascopy bid/ask ingestion, integrity checks, resampling, spread and session metadata.
2. **Features:** trend, volatility, momentum, candle geometry, market structure and liquidity proxies, calculated causally.
3. **Labels:** future-path first-touch event labels; labels are partitioned before fitting to prevent horizon overlap.
4. **Models:** gradient-boosted tabular baseline, sequence LSTM and Transformer, regime detector, calibrated meta-learner.
5. **Decision:** BUY / SELL / WAIT with calibrated probabilities, expected value, uncertainty and risk levels.
6. **Evaluation:** chronological holdout, walk-forward tests, spread/slippage/commission-aware backtests and benchmark comparisons.
7. **Serving:** FastAPI prediction service and Streamlit research dashboard.
8. **Operations:** Docker/VPS deployment, reproducible configs, monitoring and paper-trading only.

## Safety boundary

There is no broker execution module. No credentials are requested or stored. Any future live-trading integration must remain out of scope unless explicitly redesigned and separately authorized.

## Status

See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for verified, pending, and unverified work.
