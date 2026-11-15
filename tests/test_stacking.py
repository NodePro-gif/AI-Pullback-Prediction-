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


def test_stacked_training_labels_each_period_without_crossing_boundaries(
    tmp_path, monkeypatch
):
    from pullback_prediction.labels import barriers
    from pullback_prediction.training import calibration, stacking
    from pullback_prediction.data.splits import chronological_split

    train_index = pd.date_range("2015-01-01", periods=220, freq="min", tz="UTC")
    calibration_index = pd.date_range(
        "2021-01-01", periods=20, freq="min", tz="UTC"
    )
    evaluation_index = pd.date_range("2023-01-01", periods=20, freq="min", tz="UTC")
    test_index = pd.date_range("2024-01-01", periods=5, freq="min", tz="UTC")
    index = train_index.append(calibration_index).append(evaluation_index).append(test_index)
    candles = pd.DataFrame({"placeholder": 1.0}, index=index)
    partitions = chronological_split(candles)
    observed = []

    def fake_features(frame):
        return pd.DataFrame(
            {"atr_14": 1.0, "feature": np.arange(len(frame), dtype=float)},
            index=frame.index,
        )

    def fake_labels(frame, atr, **kwargs):
        observed.append(frame.index)
        return pd.Series(
            np.resize(np.array([-1.0, 0.0, 1.0]), len(frame)),
            index=frame.index,
        )

    class FakeStack:
        classes_ = np.array([-1, 0, 1])

        def predict_proba(self, frame):
            return np.tile(np.array([[0.2, 0.6, 0.2]]), (len(frame), 1))

    monkeypatch.setattr(
        "pullback_prediction.features.technical.build_features", fake_features
    )
    monkeypatch.setattr(barriers, "make_barrier_labels", fake_labels)
    monkeypatch.setattr(stacking.StackedPullbackClassifier, "fit", lambda *args, **kwargs: FakeStack())
    monkeypatch.setattr(
        calibration,
        "fit_probability_calibrators",
        lambda probabilities, labels, classes: [None, None, None],
    )
    monkeypatch.setattr(
        calibration,
        "apply_probability_calibrators",
        lambda probabilities, calibrators: probabilities,
    )

    saved = {}
    monkeypatch.setattr("joblib.dump", lambda artifact, path: saved.update(artifact=artifact))
    output = tmp_path / "stacked.joblib"
    result = stacking.train_stacked_from_frame(
        candles,
        str(output),
        n_splits=2,
        gap_bars=12,
        max_iter=5,
    )

    assert observed == [
        partitions["train"].index,
        partitions["validation"].loc[
            partitions["validation"].index < pd.Timestamp("2023-01-01", tz="UTC")
        ].index,
        partitions["validation"].loc[
            partitions["validation"].index >= pd.Timestamp("2023-01-01", tz="UTC")
        ].index,
    ]
    artifact = saved["artifact"]
    assert artifact["final_test_period"].startswith("2024-01-01")
    assert result["evaluation_rows"] == len(evaluation_index)
