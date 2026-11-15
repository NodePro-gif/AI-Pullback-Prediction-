import joblib
import numpy as np
import pandas as pd
import pytest

from pullback_prediction.features.technical import build_features
from pullback_prediction.training.inference import predict_frame


class FixedProbabilityModel:
    classes_ = np.array([-1, 0, 1])

    def predict_proba(self, features):
        return np.tile(np.array([[0.70, 0.20, 0.10]]), (len(features), 1))


def sample_candles(timeframe="M1"):
    frequency = "min" if timeframe == "M1" else "5min"
    index = pd.date_range("2024-01-01", periods=220, freq=frequency, tz="UTC")
    close = 2000 + np.arange(len(index), dtype=float) * 0.02
    frame = pd.DataFrame(index=index)
    for side, offset in (("bid", 0.0), ("ask", 0.2)):
        frame[f"{side}_open"] = close + offset
        frame[f"{side}_high"] = close + offset + 0.1
        frame[f"{side}_low"] = close + offset - 0.1
        frame[f"{side}_close"] = close + offset
    return frame


def test_predict_frame_maps_class_probabilities_to_buy_sell_wait(tmp_path):
    candles = sample_candles()
    columns = list(build_features(candles).columns)
    artifact = tmp_path / "model.joblib"
    joblib.dump(
        {"model": FixedProbabilityModel(), "feature_columns": columns},
        artifact,
    )

    result = predict_frame(candles, artifact, min_probability=0.65)

    assert result["signal"].eq(-1).all()
    assert result["predicted_class"].eq(-1).all()
    assert result["confidence"].eq(0.70).all()
    assert result["prob_sell"].eq(0.70).all()
    assert result["prob_wait"].eq(0.20).all()
    assert result["prob_buy"].eq(0.10).all()


def test_predict_frame_abstains_when_confidence_is_below_threshold(tmp_path):
    candles = sample_candles()
    columns = list(build_features(candles).columns)
    artifact = tmp_path / "model.joblib"
    joblib.dump(
        {"model": FixedProbabilityModel(), "feature_columns": columns},
        artifact,
    )

    result = predict_frame(candles, artifact, min_probability=0.80)

    assert result["signal"].eq(0).all()


def test_predict_frame_uses_saved_probability_calibrators(tmp_path):
    from pullback_prediction.training.calibration import fit_probability_calibrators

    candles = sample_candles()
    columns = list(build_features(candles).columns)
    raw_probabilities = np.array(
        [
            [0.70, 0.20, 0.10],
            [0.60, 0.30, 0.10],
            [0.10, 0.20, 0.70],
            [0.10, 0.30, 0.60],
            [0.20, 0.70, 0.10],
            [0.30, 0.60, 0.10],
        ]
    )
    calibrators = fit_probability_calibrators(
        raw_probabilities,
        np.array([-1, -1, 1, 1, 0, 0]),
        np.array([-1, 0, 1]),
    )
    artifact = tmp_path / "calibrated_model.joblib"
    joblib.dump(
        {
            "model": FixedProbabilityModel(),
            "feature_columns": columns,
            "calibrators": calibrators,
            "calibration_classes": [-1, 0, 1],
        },
        artifact,
    )

    result = predict_frame(candles, artifact, min_probability=0.0)

    assert result["probabilities_calibrated"].all()
    assert np.allclose(
        result[["prob_sell", "prob_wait", "prob_buy"]].sum(axis=1),
        1.0,
    )


def test_predict_frame_rejects_mismatched_model_timeframe(tmp_path):
    candles = sample_candles("M5")
    columns = list(build_features(candles).columns)
    artifact = tmp_path / "m1_model.joblib"
    joblib.dump(
        {
            "model": FixedProbabilityModel(),
            "feature_columns": columns,
            "timeframe": "M1",
        },
        artifact,
    )

    with pytest.raises(ValueError, match="does not match model artifact"):
        predict_frame(candles, artifact, timeframe="M5")


def test_predict_frame_rejects_candles_with_wrong_interval(tmp_path):
    candles = sample_candles()
    candles.index = pd.date_range("2024-01-01", periods=220, freq="5min", tz="UTC")
    columns = list(build_features(candles).columns)
    artifact = tmp_path / "m1_model.joblib"
    joblib.dump(
        {
            "model": FixedProbabilityModel(),
            "feature_columns": columns,
            "timeframe": "M1",
        },
        artifact,
    )

    with pytest.raises(ValueError, match="does not match"):
        predict_frame(candles, artifact, timeframe="M1")

