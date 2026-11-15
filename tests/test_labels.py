import numpy as np
import pandas as pd

from pullback_prediction.labels.barriers import make_barrier_labels


def test_barrier_label_detects_first_upper_or_lower_hit():
    index = pd.date_range("2020-01-01", periods=5, freq="min", tz="UTC")
    frame = pd.DataFrame(
        {
            "mid_close": [100, 100, 100, 100, 100],
            "mid_high": [100, 102, 100, 100, 100],
            "mid_low": [100, 99, 98, 100, 100],
        },
        index=index,
    )
    atr = pd.Series([1.0] * 5, index=index)
    labels = make_barrier_labels(frame, atr, horizon_bars=2, barrier_atr_multiple=1.5)
    assert labels.iloc[0] == 1
    assert labels.iloc[1] == -1


def test_barrier_label_censors_last_horizon_rows():
    index = pd.date_range("2020-01-01", periods=5, freq="min", tz="UTC")
    frame = pd.DataFrame(
        {
            "mid_close": [100] * 5,
            "mid_high": [100] * 5,
            "mid_low": [100] * 5,
        },
        index=index,
    )
    atr = pd.Series([1.0] * 5, index=index)
    labels = make_barrier_labels(frame, atr, horizon_bars=2)
    assert labels.iloc[:3].eq(0).all()
    assert labels.iloc[3:].isna().all()
