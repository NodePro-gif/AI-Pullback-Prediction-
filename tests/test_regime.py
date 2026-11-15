import numpy as np
import pandas as pd
import pytest

from pullback_prediction.training.regime import (
    DEFAULT_REGIME_FEATURES,
    RegimeDetector,
)


def _regime_features(rows=120):
    rng = np.random.default_rng(19)
    index = pd.date_range("2020-01-01", periods=rows, freq="min", tz="UTC")
    values = rng.normal(size=(rows, len(DEFAULT_REGIME_FEATURES)))
    values[:, 0] = np.exp(values[:, 0]) * 0.001
    values[:, 1] = np.abs(values[:, 1]) * 0.0001
    values[:, 2] = np.abs(values[:, 2]) * 0.0002
    return pd.DataFrame(values, index=index, columns=DEFAULT_REGIME_FEATURES)


def test_regime_detector_assigns_stable_clusters_and_keeps_warmup_nan():
    features = _regime_features()
    features.iloc[:5] = np.nan
    detector = RegimeDetector.fit(features, n_regimes=3)
    first = detector.transform(features)
    second = detector.transform(features)

    pd.testing.assert_frame_equal(first, second)
    assert first.iloc[:5].isna().all().all()
    assert first.iloc[5:]["regime_id"].between(0, 2).all()
    assert first.iloc[5:]["regime_distance"].ge(0).all()


def test_regime_detector_fit_rejects_insufficient_or_duplicate_rows():
    features = _regime_features(20)
    with pytest.raises(ValueError, match="at least 2"):
        RegimeDetector.fit(features, n_regimes=1)

    duplicate = pd.DataFrame(
        np.ones((20, len(DEFAULT_REGIME_FEATURES))),
        index=features.index,
        columns=DEFAULT_REGIME_FEATURES,
    )
    with pytest.raises(ValueError, match="unique rows"):
        RegimeDetector.fit(duplicate, n_regimes=2)


def test_regime_detector_rejects_missing_feature_columns():
    features = _regime_features().drop(columns=[DEFAULT_REGIME_FEATURES[0]])
    with pytest.raises(ValueError, match="Missing regime features"):
        RegimeDetector.fit(features)
