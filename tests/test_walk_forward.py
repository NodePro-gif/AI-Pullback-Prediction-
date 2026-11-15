import numpy as np
import pandas as pd
import pytest

from pullback_prediction.training.walk_forward import walk_forward_evaluate


def make_data(rows=180):
    index = pd.date_range("2020-01-01", periods=rows, freq="h", tz="UTC")
    rng = np.random.default_rng(19)
    features = pd.DataFrame(
        {"signal": rng.normal(size=rows), "noise": rng.normal(size=rows)},
        index=index,
    )
    labels = pd.Series(np.where(features["signal"] > 0, 1, -1), index=index)
    return features, labels


def test_walk_forward_uses_chronological_expanding_folds_and_gap():
    features, labels = make_data()
    result = walk_forward_evaluate(
        features, labels, n_splits=4, gap_bars=7, max_iter=15
    )

    folds = result["folds"]
    assert len(folds) == 4
    assert result["summary"]["folds"] == 4
    assert result["summary"]["test_rows"] == sum(f["test_rows"] for f in folds)
    for previous, current in zip(folds, folds[1:]):
        assert previous["test_end"] < current["test_start"]
        assert previous["train_rows"] < current["train_rows"]
    assert all(0.0 <= fold["accuracy"] <= 1.0 for fold in folds)


def test_walk_forward_gap_is_applied_before_invalid_rows_are_removed():
    features, labels = make_data(90)
    labels.iloc[30:35] = np.nan
    features.iloc[20, 0] = np.inf

    result = walk_forward_evaluate(
        features, labels, n_splits=3, gap_bars=5, max_iter=10
    )

    assert result["summary"]["test_rows"] < 90
    assert all(fold["train_rows"] > 0 and fold["test_rows"] > 0 for fold in result["folds"])


def test_walk_forward_rejects_unaligned_indexes():
    features, labels = make_data()
    with pytest.raises(ValueError, match="identical indexes"):
        walk_forward_evaluate(features, labels.reset_index(drop=True))


def test_walk_forward_rejects_non_chronological_index():
    features, labels = make_data()
    features = features.iloc[::-1]
    labels = labels.iloc[::-1]
    with pytest.raises(ValueError, match="chronological"):
        walk_forward_evaluate(features, labels)
