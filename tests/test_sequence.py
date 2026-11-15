import numpy as np
import pandas as pd
import pytest

from pullback_prediction.training.sequence import make_sequence_windows


def test_sequence_windows_use_only_trailing_rows_and_preserve_end_timestamps():
    index = pd.date_range("2020-01-01", periods=8, freq="h", tz="UTC")
    features = pd.DataFrame({"x": np.arange(8), "y": np.arange(8) * 2}, index=index)
    labels = pd.Series(np.arange(8) % 3 - 1, index=index)

    windows = make_sequence_windows(features, labels, sequence_length=3)

    assert windows.features.shape == (6, 3, 2)
    assert windows.labels.tolist() == labels.iloc[2:].tolist()
    assert windows.timestamps.equals(index[2:])
    np.testing.assert_array_equal(windows.features[0, :, 0], [0, 1, 2])


def test_sequence_windows_drop_any_window_with_invalid_features_or_label():
    index = pd.date_range("2020-01-01", periods=8, freq="h", tz="UTC")
    features = pd.DataFrame({"x": np.arange(8, dtype=float)}, index=index)
    labels = pd.Series(np.ones(8), index=index)
    features.iloc[3, 0] = np.inf
    labels.iloc[6] = np.nan

    windows = make_sequence_windows(features, labels, sequence_length=3)

    assert index[3] not in windows.timestamps
    assert index[4] not in windows.timestamps
    assert index[6] not in windows.timestamps


def test_sequence_windows_reject_misaligned_indexes():
    index = pd.date_range("2020-01-01", periods=8, freq="h", tz="UTC")
    features = pd.DataFrame({"x": np.arange(8)}, index=index)
    labels = pd.Series(np.ones(8), index=index[::-1])
    with pytest.raises(ValueError, match="identical indexes"):
        make_sequence_windows(features, labels)
