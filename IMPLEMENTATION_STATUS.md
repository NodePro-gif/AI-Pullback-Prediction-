# Implementation status

Statuses: **implemented, unverified** means code is in the repository but has not yet been executed in CI; **verified** requires a passing recorded check; **pending** means not implemented.

| Component | Status | Evidence / next check |
|---|---|---|
| Repository and project contract | implemented, unverified | README and hard constraints committed |
| Python package metadata and dependencies | implemented, unverified | CI install on Python 3.11 and 3.12 |
| CLI for data validation and splitting | implemented, unverified | CI import; CLI smoke test still required |
| Dukascopy bid/ask CSV loader | implemented, unverified | Unit tests in tests/test_loader.py |
| Fixed chronological partitions | implemented, unverified | Boundary tests in tests/test_splits.py |
| Causal technical/microstructure features | implemented, unverified | Tests include future-perturbation leakage check |
| ATR first-touch barrier labels | implemented, unverified | First-touch, ambiguous-bar and censoring tests |
| Automated CI | implemented, unverified | GitHub Actions run must be inspected |
| Paired Dukascopy bid/ask file ingestion | pending | Needed if provider exports bid and ask as separate files |
| XGBoost baseline | pending | Train on train partition; evaluate on validation |
| LSTM and Transformer | pending | Shape, masking, reproducibility tests |
| Regime detector and meta-learner | pending | Walk-forward calibration tests |
| Probability calibration and WAIT policy | pending | Reliability and abstention tests |
| Cost-aware backtester | pending | Hand-calculated bid/ask trade fixtures |
| FastAPI service | pending | API contract tests |
| Streamlit dashboard | pending | Launch and smoke test |
| Docker/VPS deployment | pending | Container build and health check |

## Hard constraints

- Instrument: XAUUSD.
- Timeframes: M1 and M5.
- Canonical data: Dukascopy bid and ask.
- Training: 2015–2020.
- Validation: 2021–2023.
- Final test: 2024 onward; do not use for tuning.
- Outputs: BUY / SELL / WAIT.
- No broker order execution; live trading remains disabled.
