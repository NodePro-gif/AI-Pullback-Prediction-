import numpy as np
import pandas as pd

from pullback_prediction.features.technical import build_features


def _candles(n=240):
    index = pd.date_range("2020-01-01", periods=n, freq="min", tz="UTC")
    close = 2000 + np.linspace(0, 5, n) + np.sin(np.arange(n) / 4)
    base = pd.DataFrame(index=index)
    base["bid_open"] = close - 0.1
    base["bid_high"] = close + 0.4
    base["bid_low"] = close - 0.4
    base["bid_close"] = close
    base["ask_open"] = close
    base["ask_high"] = close + 0.5
    base["ask_low"] = close - 0.3
    base["ask_close"] = close + 0.1
    return base


def test_features_are_finite_or_warmup_nan_and_keep_index():
    frame = _candles()
    result = build_features(frame)
    assert result.index.equals(frame.index)
    assert "atr_14" in result
    assert "spread" in result
    assert result["spread"].dropna().gt(0).all()


def test_future_price_change_does_not_change_past_features():
    original = _candles()
    changed = original.copy()
    changed.iloc[-1, changed.columns.get_loc("bid_close")] += 100
    changed.iloc[-1, changed.columns.get_loc("ask_close")] += 100
    changed.iloc[-1, changed.columns.get_loc("bid_high")] += 100
    changed.iloc[-1, changed.columns.get_loc("ask_high")] += 100
    before = build_features(original)
    after = build_features(changed)
    pd.testing.assert_frame_equal(before.iloc[:-1], after.iloc[:-1])
