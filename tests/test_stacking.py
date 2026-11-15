import numpy as np
import pandas as pd
import pytest

from pullback_prediction.training.stacking import StackedPullbackClassifier


def _stacking_data(rows=360):
    rng = np.random.default_rng(23)
    index = pd.date_range("2015-01-01", periods=rows, freq="min", tz="UTC")
    latent = rng.normal(size=(rows, 5))
    labels = np.where(
        latent[:, 0] > 0.45,
        1,
        np.where(latent[:, 0] < -0.45, -1, 0),
    )
    features = pd.DataFrame(
        latent,
        index=index,
        columns=[f"feature_{i}" for i in range(latent.shape[1])],
    )
    return features, pd.Series(labels, index=index, name="target")


def test_stacked_classifier_trains_oof_meta_and_returns_simplex_probabilities():
    features, labels = _stacking_data()
    features.iloc[:8, 0] = np.nan
    model = StackedPullbackClassifier.fit(
        features,
        labels,
        n_splits=4,
        gap_bars=8,
        max_iter=20,
    )
    probabilities = model.predict_proba(features.iloc[-20:])

    assert probabilities.shape == (20, 3)
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-8)
    assert model.classes_.tolist() == [-1, 0, 1]


def test_stacked_classifier_rejects_misaligned_or_nonchronological_data():
    features, labels = _stacking_data()
    with pytest.raises(ValueError, match="identical indexes"):
        StackedPullbackClassifier.fit(features, labels.iloc[::-1])

    reversed_features = features.iloc[::-1]
    with pytest.raises(ValueError, match="chronological"):
        StackedPullbackClassifier.fit(reversed_features, labels.iloc[::-1])


def test_stacked_classifier_rejects_feature_schema_changes():
    features, labels = _stacking_data()
    model = StackedPullbackClassifier.fit(
        features,
        labels,
        n_splits=3,
        gap_bars=5,
        max_iter=10,
    )
    with pytest.raises(ValueError, match="feature columns"):
        model.predict_proba(features.iloc[-5:].drop(columns=["feature_0"]))
