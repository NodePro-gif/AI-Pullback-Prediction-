"""Chronological train, validation and untouched final-test splits."""

from __future__ import annotations

import pandas as pd

TRAIN_START = pd.Timestamp("2015-01-01", tz="UTC")
VALIDATION_START = pd.Timestamp("2021-01-01", tz="UTC")
TEST_START = pd.Timestamp("2024-01-01", tz="UTC")


def chronological_split(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split indexed candles into fixed UTC date windows without shuffling."""
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("Input frame must use a DatetimeIndex.")
    if frame.index.tz is None:
        raise ValueError("DatetimeIndex must be timezone-aware.")
    index = frame.index.tz_convert("UTC")
    if not index.is_monotonic_increasing:
        raise ValueError("Input timestamps must be sorted ascending.")
    if index.has_duplicates:
        raise ValueError("Input timestamps must be unique.")

    train = frame.loc[(index >= TRAIN_START) & (index < VALIDATION_START)].copy()
    validation = frame.loc[(index >= VALIDATION_START) & (index < TEST_START)].copy()
    test = frame.loc[index >= TEST_START].copy()
    return {"train": train, "validation": validation, "test": test}
