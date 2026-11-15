# Implementation status

Statuses: **verified** means the stated check was run and passed; **implemented, unverified** means code was written but has not been executed in a matching environment; **pending** means not implemented.

| Component | Status | Evidence / next check |
|---|---|---|
| Repository and project contract | implemented, unverified | Review README and constraints |
| Python package and CLI | pending | Install package; run CLI help |
| Dukascopy bid/ask CSV loader | pending | Unit tests with valid and malformed fixtures |
| Chronological data partitions | pending | Assert date boundaries and no overlap |
| Causal feature engineering | pending | Unit tests plus future-perturbation leakage test |
| Pullback event labels | pending | Boundary and future-window tests |
| XGBoost baseline | pending | Train on train partition; evaluate on validation |
| LSTM and Transformer | pending | Shape, masking, and reproducibility tests |
| Regime detector and meta-learner | pending | Walk-forward calibration tests |
| Probability calibration and WAIT policy | pending | Reliability and abstention tests |
| Cost-aware backtester | pending | Hand-calculated trade fixtures |
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
