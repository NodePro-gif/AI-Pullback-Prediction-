"""Forward outcome labels. This module is for targets, never model features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_barrier_labels(
    frame: pd.DataFrame,
    atr: pd.Series,
    horizon_bars: int = 12,
    barrier_atr_multiple: float = 1.5,
) -> pd.Series:
    """Label which ATR barrier is reached first after each candle close.

    Returns int8-compatible labels: +1 upper barrier first, -1 lower barrier
    first, 0 neither barrier within horizon. NaN marks insufficient future data
    or a candle where both barriers are touched and intrabar ordering is unknown.

    Bid/ask execution is not simulated here; this is a supervised outcome label.
    Costs belong in the backtester.
    """
    if horizon_bars < 1:
        raise ValueError("horizon_bars must be at least 1.")
    if barrier_atr_multiple <= 0:
        raise ValueError("barrier_atr_multiple must be positive.")
    required = {"mid_close", "mid_high", "mid_low"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    if not frame.index.equals(atr.index):
        raise ValueError("ATR index must exactly match the candle index.")

    close = frame["mid_close"].to_numpy(dtype=float)
    high = frame["mid_high"].to_numpy(dtype=float)
    low = frame["mid_low"].to_numpy(dtype=float)
    atr_values = atr.to_numpy(dtype=float)
    labels = np.full(len(frame), np.nan, dtype=float)

    for i in range(len(frame) - horizon_bars):
        if not np.isfinite(close[i]) or not np.isfinite(atr_values[i]) or atr_values[i] <= 0:
            continue
        upper = close[i] + barrier_atr_multiple * atr_values[i]
        lower = close[i] - barrier_atr_multiple * atr_values[i]
        label = 0.0
        ambiguous = False
        for j in range(i + 1, i + horizon_bars + 1):
            hit_upper = high[j] >= upper
            hit_lower = low[j] <= lower
            if hit_upper and hit_lower:
                ambiguous = True
                break
            if hit_upper:
                label = 1.0
                break
            if hit_lower:
                label = -1.0
                break
        if not ambiguous:
            labels[i] = label

    return pd.Series(labels, index=frame.index, name="target", dtype="float64")
