import pandas as pd
import pytest

from pullback_prediction.data.splits import chronological_split


def _frame():
    index = pd.to_datetime(
        [
            "2015-01-01T00:00:00Z",
            "2020-12-31T23:59:00Z",
            "2021-01-01T00:00:00Z",
            "2023-12-31T23:59:00Z",
            "2024-01-01T00:00:00Z",
            "2026-01-01T00:00:00Z",
        ],
        utc=True,
    )
    return pd.DataFrame({"close": range(len(index))}, index=index)


def test_split_respects_fixed_inclusive_and_exclusive_boundaries():
    result = chronological_split(_frame())
    assert result["train"]["close"].tolist() == [0, 1]
    assert result["validation"]["close"].tolist() == [2, 3]
    assert result["test"]["close"].tolist() == [4, 5]


def test_split_rejects_naive_timestamps():
    frame = pd.DataFrame(
        {"close": [1]},
        index=pd.DatetimeIndex(["2020-01-01"]),
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        chronological_split(frame)


def test_split_rejects_unsorted_input():
    frame = _frame().iloc[::-1]
    with pytest.raises(ValueError, match="sorted ascending"):
        chronological_split(frame)
