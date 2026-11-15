"""Causal feature engineering for bid/ask OHLC candles."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    result = 100 - (100 / (1 + relative_strength))
    return result.where(loss.ne(0), 100.0).where(gain.ne(0) | loss.ne(0))


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build features using current and past candles only.

    Required input columns are bid/ask OHLC. Output retains the original index
    and contains no forward-looking values. Warm-up NaNs are intentional.
    """
    required = {
        "bid_open", "bid_high", "bid_low", "bid_close",
        "ask_open", "ask_high", "ask_low", "ask_close",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing bid/ask columns: {', '.join(sorted(missing))}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("Input must have a DatetimeIndex.")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("Input timestamps must be unique and sorted ascending.")

    result = pd.DataFrame(index=frame.index)
    for field in ("open", "high", "low", "close"):
        result[f"mid_{field}"] = (frame[f"bid_{field}"] + frame[f"ask_{field}"]) / 2
    result["spread"] = frame["ask_close"] - frame["bid_close"]
    result["spread_bps"] = result["spread"] / result["mid_close"] * 10_000

    close = result["mid_close"]
    high = result["mid_high"]
    low = result["mid_low"]
    previous_close = close.shift(1)

    for lag in (1, 2, 3, 5, 10, 20):
        result[f"return_{lag}"] = close.pct_change(lag)

    for period in (20, 50, 200):
        ema = close.ewm(span=period, adjust=False, min_periods=period).mean()
        result[f"ema_{period}"] = ema
        result[f"close_to_ema_{period}"] = close / ema - 1

    result["rsi_14"] = _rsi(close, 14)
    ema_fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_slow = close.ewm(span=26, adjust=False, min_periods=26).mean()
    result["macd"] = ema_fast - ema_slow
    result["macd_signal"] = result["macd"].ewm(
        span=9, adjust=False, min_periods=9
    ).mean()
    result["macd_histogram"] = result["macd"] - result["macd_signal"]

    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    result["atr_14"] = true_range.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()
    result["atr_pct"] = result["atr_14"] / close

    for window in (5, 15, 30, 60):
        result[f"volatility_{window}"] = close.pct_change().rolling(
            window, min_periods=window
        ).std()
        result[f"range_atr_{window}"] = (
            (high.rolling(window, min_periods=window).max()
             - low.rolling(window, min_periods=window).min())
            / result["atr_14"]
        )

    candle_range = (high - low).replace(0, np.nan)
    result["body_to_range"] = (close - result["mid_open"]) / candle_range
    result["upper_wick_to_range"] = (
        high - pd.concat([result["mid_open"], close], axis=1).max(axis=1)
    ) / candle_range
    result["lower_wick_to_range"] = (
        pd.concat([result["mid_open"], close], axis=1).min(axis=1) - low
    ) / candle_range

    if "volume" in frame.columns:
        volume = pd.to_numeric(frame["volume"], errors="coerce")
        result["log_volume"] = np.log1p(volume.clip(lower=0))
        volume_mean = volume.rolling(30, min_periods=30).mean()
        volume_std = volume.rolling(30, min_periods=30).std()
        result["volume_zscore_30"] = (volume - volume_mean) / volume_std.replace(0, np.nan)

    result["hour_utc"] = result.index.hour
    result["day_of_week_utc"] = result.index.dayofweek
    return result.replace([np.inf, -np.inf], np.nan)
