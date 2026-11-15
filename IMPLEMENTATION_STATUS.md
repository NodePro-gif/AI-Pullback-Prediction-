# Implementation status

Statuses: **verified** means passing recorded checks; **implemented, unverified** means code exists but runtime or data-scale behavior has not yet been validated; **pending** means not implemented.

| Component | Status | Evidence / next check |
|---|---|---|
| Repository and project contract | verified | README and hard constraints committed |
| Python package metadata and dependencies | implemented, unverified | CI dependency installation |
| CLI for validation, splitting, and baseline training | implemented, unverified | CI plus CLI smoke test |
| Cost-aware bid/ask backtest engine | verified | Hand-calculated long/short, slippage, commission, and conservative same-bar stop tests pass in CI |
| Backtest CLI | implemented, unverified | CLI smoke test and real-data run pending |
| Dukascopy bid/ask CSV loader | verified | Unit tests pass in CI |
| Separate Dukascopy bid/ask file ingestion | verified | Paired-file alignment tests pass in CI |
| Fixed chronological partitions | verified | Boundary tests pass in CI |
| Causal technical/microstructure features | verified | Future-perturbation leakage test passes in CI |
| ATR first-touch barrier labels | verified | First-touch, ambiguous-bar and censoring tests pass in CI |
| HistGradientBoosting baseline | implemented, unverified | Train-only fitting and validation metrics; CI pending |
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
