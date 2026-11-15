# Implementation status

Statuses: **verified** means passing recorded checks; **implemented, unverified** means code exists but runtime or data-scale behavior has not yet been validated; **pending** means not implemented.

| Component | Status | Evidence / next check |
|---|---|---|
| Repository and project contract | verified | README and hard constraints committed |
| Python package metadata and dependencies | implemented, unverified | CI dependency installation |
| CLI for downloading, validation, Parquet preparation, splitting, training, prediction, walk-forward evaluation, and backtesting | implemented, unverified | Prepare-data and backtest smoke tests pass; walk-forward CLI test pending |
| Cost-aware bid/ask backtest engine | verified | Long/short fills, costs, ambiguous bars, and drawdown tests pass in CI |
| Backtest CLI | verified | Parquet-to-trades/summary smoke test passes; real-data run pending |
| Dukascopy bid/ask CSV loader | verified | Unit tests pass in CI, including Unix-millisecond timestamps |
| Dukascopy bid/ask downloader | implemented, unverified | Pinned 1.51.1 CLI arguments, cache/retry flags, and no-overwrite tests pass; live network download pending |
| Bid/ask CSV to Parquet preparation | verified | CLI smoke test validates numeric-millisecond timestamps and writes UTC Parquet |
| Separate Dukascopy bid/ask file ingestion | verified | Paired-file alignment tests pass in CI |
| Fixed chronological partitions | verified | Boundary tests pass in CI |
| Causal technical/microstructure features | verified | Future-perturbation leakage test passes in CI |
| ATR first-touch barrier labels | verified | First-touch, ambiguous-bar and censoring tests pass in CI |
| HistGradientBoosting baseline | implemented, unverified at data scale | Fit on 2015–2020; full Dukascopy training run pending |
| Model inference and confidence abstention | verified | Calibrated probability mapping and abstention tests pass in CI |
| XGBoost baseline | pending | Later model comparison candidate |
| LSTM and Transformer | pending | Sequence windows, masks, reproducibility and walk-forward checks |
| Regime detector and meta-learner | pending | Walk-forward calibration tests |
| Isotonic probability calibration | verified | Fit on 2021–2022 and evaluate on 2023; probability simplex and inference tests pass |
| Calibrated WAIT/abstention policy | implemented, unverified at data scale | Configurable confidence threshold; calibration and backtest analysis on real data pending |
| Expanding-window walk-forward evaluation | verified | Chronological TimeSeriesSplit folds, candle-based leakage gap and unit tests pass |
| Walk-forward CLI | implemented, unverified | CLI test excludes 2024+ final test rows; latest CI pending |
| Cost-aware backtester | pending | Hand-calculated bid/ask trade fixtures |
| FastAPI service | pending | API contract tests |
| Streamlit dashboard | pending | Launch and smoke test |
| Docker/VPS deployment | pending | Container build and health check |

## Hard constraints

- Instrument: XAUUSD.
- Timeframes: M1 and M5.
- Canonical data: Dukascopy bid and ask.
- Training: 2015–2020 only.
- Validation: 2021–2023 only.
- Final test: 2024 onward; excluded from training and model selection.
- Outputs: BUY / SELL / WAIT.
- No broker order execution; live trading remains disabled.
