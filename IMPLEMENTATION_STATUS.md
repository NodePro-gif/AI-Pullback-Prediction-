# Implementation status

Statuses: **verified** means passing recorded checks; **implemented, unverified** means code exists but runtime or data-scale behavior has not yet been validated; **pending** means not implemented.

| Component | Status | Evidence / next check |
|---|---|---|
| Repository and project contract | verified | README and hard constraints committed |
| Python package metadata and dependencies | implemented, unverified | CI dependency installation |
| CLI for downloading, validation, Parquet preparation, splitting, training, prediction, and backtesting | implemented, unverified | Prepare-data and backtest smoke tests pass; full workflow smoke test pending |
| Cost-aware bid/ask backtest engine | verified | Long/short fills, costs, ambiguous bars, and drawdown tests pass in CI |
| Backtest CLI | verified | Parquet-to-trades/summary smoke test passes; real-data run pending |
| Dukascopy bid/ask CSV loader | verified | Unit tests pass in CI, including Unix-millisecond timestamps |
| Dukascopy bid/ask downloader | implemented, unverified | Pinned 1.51.1 CLI arguments, cache/retry flags, and no-overwrite tests pass; live network download pending |
| Bid/ask CSV to Parquet preparation | verified | CLI smoke test validates numeric-millisecond timestamps and writes UTC Parquet |
| Separate Dukascopy bid/ask file ingestion | verified | Paired-file alignment tests pass in CI |
| Fixed chronological partitions | verified | Boundary tests pass in CI |
| Causal technical/microstructure features | verified | Future-perturbation leakage test passes in CI |
| ATR first-touch barrier labels | verified | First-touch, ambiguous-bar and censoring tests pass in CI |
| HistGradientBoosting baseline | verified | 14 tests and Ruff pass on Python 3.11/3.12; full market-data training run pending |
| Model inference and confidence abstention | verified | Class-probability mapping and abstention tests pass in CI; scores remain uncalibrated |
| XGBoost baseline | pending | Later model comparison candidate |
| LSTM and Transformer | pending | Sequence windows, masks, reproducibility and walk-forward checks |
| Regime detector and meta-learner | pending | Walk-forward calibration tests |
| Probability calibration and WAIT policy | pending | Reliability, expected-value and abstention tests |
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
