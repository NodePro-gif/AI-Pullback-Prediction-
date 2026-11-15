# Implementation status

Statuses: **verified** means passing recorded checks; **implemented, unverified** means code exists but runtime or data-scale behavior has not yet been validated; **pending** means not implemented.

| Component | Status | Evidence / next check |
|---|---|---|
| Repository contract and chronological data policy | verified | README and fixed date boundaries committed |
| Dukascopy bid/ask loader and downloader | implemented, unverified at scale | CSV, timestamp, alignment, retry/cache tests pass; full historical download pending |
| Parquet preparation and chronological splitting | verified | CLI conversion and boundary tests pass |
| Causal technical features and ATR first-touch labels | verified | Future-perturbation, first-touch, ambiguous-bar and censoring tests; EMA trend strength, ATR/volatility ratios, spread context and cyclical UTC session features |
| HistGradientBoosting baseline | verified in unit tests | Full 2015–2020 market-data fit pending |
| Isotonic calibration and inference abstention | verified in unit tests | Calibration on 2021–2022; holdout evaluation on 2023 |
| Expanding-window walk-forward | verified in unit tests | Candle gap enforced before invalid-row filtering; final-test partition excluded |
| HistGradientBoosting benchmark | verified in CI | Same expanding folds and metrics; test coverage on Python 3.11/3.12 |
| XGBoost benchmark path | implemented, unverified | Optional dependency and binary/multiclass mapping; run with `.[xgboost]` on real data |
| Bid/ask cost-aware backtester | verified in unit tests | Next-open entries, adverse slippage, commission, stop-first collisions |
| CLI workflow | verified by smoke/unit tests | Download, validate, prepare, split, train, predict, walk-forward, benchmark, backtest; full historical run pending |
| FastAPI inference service | verified by contract tests | Read-only /health and /predict; timestamp, OHLC, quote validation; no order execution |
| Streamlit research dashboard | implemented, verification pending | Local parquet upload and model inference UI |
| Docker image | implemented, verification pending | API container; artifact must be mounted at runtime |
| Sequence window generation | verified by unit tests | Chronological trailing windows; invalid windows dropped |
| LSTM / Transformer training | implemented, unverified | CUDA/MPS auto-selection, CUDA AMP, batched validation, class weighting, gradient clipping and early stopping; PyTorch runtime test is optional and skipped in base CI; fold integration and inference export pending |
| Regime detector and meta-learner | pending | Requires out-of-fold predictions and separate calibration |
| Real-data benchmark and untouched final test | pending | Requires complete Dukascopy M1/M5 data and compute run |

## Hard constraints

- Instrument: XAUUSD; timeframes M1 and M5.
- Canonical data: Dukascopy bid and ask.
- Training: 2015–2020 only.
- Calibration: 2021–2022; validation/evaluation: 2023.
- Final test: 2024 onward; never used for training, calibration or model selection.
- Output: BUY / SELL / WAIT.
- No broker credentials, order execution or live trading.
- Report measured results only; never promise zero loss, zero perplexity, or a guaranteed win rate.
