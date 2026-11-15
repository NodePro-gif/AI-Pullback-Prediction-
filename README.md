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

## Download XAUUSD history

The downloader requires Node.js/npm (`npx`). It pins `dukascopy-node` 1.51.1, downloads bid and ask separately, keeps a disk cache for interrupted runs, and refuses to overwrite completed raw files. The end date is exclusive.

```bash
pullback-predict download-data --from-date 2015-01-01 --to-date 2021-01-01 --timeframe M1 --output-dir data/raw
```

Repeat for `2021-01-01` to `2024-01-01` (validation) and `2024-01-01` to the desired end date (final test), and repeat for M5. Validate a bid/ask pair:

```bash
pullback-predict validate-data --input data/raw/xauusd-m1-bid-2015-01-01-2021-01-01.csv --ask-input data/raw/xauusd-m1-ask-2015-01-01-2021-01-01.csv --timeframe M1
```

Dukascopy exports Unix timestamps in milliseconds; the loader normalizes them to UTC.

Validate a downloaded Dukascopy CSV:

```bash
pullback-predict validate-data --input data/raw/XAUUSD_M1.csv --timeframe M1
```

Combine the two validated exports into a processed Parquet file:

```bash
pullback-predict prepare-data --bid-input data/raw/xauusd-m1-bid-2015-01-01-2021-01-01.csv --ask-input data/raw/xauusd-m1-ask-2015-01-01-2021-01-01.csv --timeframe M1 --output data/processed/XAUUSD_M1.parquet
```

The command refuses to overwrite an existing processed file. Create chronological partitions:

```bash
pullback-predict split-data --input data/processed/XAUUSD_M1.parquet --output-dir data/splits
```

Train the baseline on 2015–2020 and evaluate on 2021–2023:

```bash
pullback-predict train-baseline --input data/processed/XAUUSD_M1.parquet --output artifacts/xauusd_m1_baseline.joblib --timeframe M1
```

The command fits on 2015–2020, fits one-vs-rest isotonic calibrators on 2021–2022, and reports holdout metrics on 2023. It writes a model artifact and sibling `.metrics.json`. The 2024+ final-test partition is not accessed. Run separately for M5 data with `--timeframe M5` and a distinct artifact path. Training artifacts record their timeframe; prediction, dashboard, and API reject mismatched candle intervals or artifacts.

Run expanding-window walk-forward evaluation on development data (2015–2023). The horizon-sized gap is enforced before invalid rows are removed, and 2024+ is excluded:

```bash
pullback-predict walk-forward --input data/processed/XAUUSD_M1.parquet --output-dir artifacts/walk_forward_m1 --n-splits 5 --horizon-bars 12
```

The command writes fold-level metrics and an aggregate `summary.json`.

Compare HistGradientBoosting and optional XGBoost using the same chronological folds:

```bash
pip install -e ".[xgboost]"
pullback-predict benchmark-models --input data/processed/XAUUSD_M1.parquet --output-dir artifacts/benchmark_m1 --n-splits 5 --horizon-bars 12
```

The benchmark writes per-model fold CSVs and aggregate metrics. Install `.[xgboost]` for XGBoost without installing PyTorch; install `.[deep-learning]` separately for PyTorch sequence models. XGBoost is skipped with a recorded reason if the optional dependency is not installed. Model comparisons use only 2015–2023 development data; the 2024+ final test is not opened.

Fit unsupervised market regimes using only the 2015–2020 training partition. The saved scaler and cluster centers are then applied to all rows without refitting on validation or final-test data:

```bash
pullback-predict fit-regimes --input data/processed/XAUUSD_M1.parquet --output-dir artifacts/regimes_m1 --n-regimes 4
```

The command writes a reusable detector artifact and a parquet file with `regime_id` and `regime_distance`. Regime IDs are cluster identifiers, not guaranteed bullish/bearish labels.

Train a leakage-aware stacked classifier. Base-model probabilities for the meta-learner are generated only from expanding-window out-of-fold predictions with a horizon-sized gap; the final base model is refit on 2015–2020, calibrated on 2021–2022, and evaluated on 2023:

```bash
pullback-predict train-stacked --input data/processed/XAUUSD_M1.parquet --output artifacts/xauusd_stacked.joblib --timeframe M1 --n-splits 5 --horizon-bars 12
pullback-predict predict --input data/processed/XAUUSD_M1.parquet --model artifacts/xauusd_stacked.joblib --output data/processed/XAUUSD_M1_stacked_signals.parquet --timeframe M1
```

Train an end-to-end LSTM or Transformer sequence model. The pipeline fits its feature scaler on training rows only, trains on 2015–2020, uses 2021 for early stopping and 2022 for probability calibration, and reports 2023 evaluation metrics. It never opens 2024+ final-test rows. PyTorch is optional:

```bash
pip install -e ".[deep-learning]"
pullback-predict train-sequence --input data/processed/XAUUSD_M1.parquet --output artifacts/xauusd_m1_lstm.joblib --architecture lstm --timeframe M1 --sequence-length 128 --epochs 30 --batch-size 512
```

Use `--architecture transformer` for a Transformer. On a CUDA runtime (including a compatible Kaggle GPU session), pass `--device cuda`; CUDA mixed precision is enabled by default and can be disabled with `--no-mixed-precision`. Without CUDA, the trainer selects Apple MPS when available, otherwise CPU. Batch size and sequence length should be adjusted to available GPU memory.

Each run writes a joblib metadata artifact, a sibling `.pt` model checkpoint, and a `.metrics.json` report. Keep the two model files together. Generate BUY/SELL/WAIT predictions with the same artifact:

```bash
pullback-predict predict --input data/processed/XAUUSD_M1.parquet --model artifacts/xauusd_m1_lstm.joblib --output data/processed/XAUUSD_M1_lstm_signals.parquet --timeframe M1
```

Sequence inference aligns each prediction to the final candle in its trailing window; the initial warm-up rows abstain with WAIT. CPU PyTorch sequence and inference tests run in a dedicated CI job. GPU execution and full-history model quality still require measurement on the target runtime and real Dukascopy data.

## Research dashboard and API

Install the optional interfaces:

```bash
pip install -e ".[api,dashboard]"
streamlit run dashboard/app.py
```

Start the read-only API after placing a trained artifact at the configured path:

```bash
PULLBACK_MODEL_PATH=artifacts/xauusd_baseline.joblib uvicorn pullback_prediction.api.app:app --host 0.0.0.0 --port 8000
```

The API exposes `GET /health` and `POST /predict`; it has no broker or order endpoint. Build and run the container with the model artifact mounted read-only:

```bash
docker build -t pullback-api .
docker run --rm -p 8000:8000 -v "$PWD/artifacts:/app/artifacts:ro" pullback-api
```

Do not expose the research API publicly without adding authentication, TLS, rate limits, and operational monitoring.

Generate out-of-sample or validation predictions from a saved model. New artifacts apply the 2021–2022 isotonic calibrators before the abstention threshold; older artifacts without calibrators use raw model scores:

```bash
pullback-predict predict --input data/processed/XAUUSD_M1.parquet --model artifacts/xauusd_baseline.joblib --output data/processed/XAUUSD_M1_signals.parquet --min-probability 0.60 --timeframe M1
```

Then backtest the resulting parquet, which contains `signal` (-1 short, 0 wait, +1 long) and `atr_14` alongside bid/ask OHLC:

```bash
pullback-predict backtest-signals --input data/processed/XAUUSD_M1_signals.parquet --output-dir artifacts/backtest_m1 --slippage-price 0.05
```

The simulator enters at the next candle open, uses ask for long entries and bid for short entries, and exits against the opposite quote. Stop/target collisions within one candle are resolved conservatively as stop-loss first. PnL is price units per unit, not account currency or lot sizing.

## Planned architecture

1. **Data:** Dukascopy bid/ask ingestion, integrity checks, resampling, spread and session metadata.
2. **Features:** trend strength across EMA horizons, ATR-normalized trend, multi-horizon realized volatility, ATR changes, bid/ask spread z-scores and spread-to-ATR, candle geometry, volume context when available, and cyclical UTC session features. All are calculated causally from current and past candles.
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
