"""Model inference and confidence-based abstention."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from pullback_prediction.features.technical import build_features
from pullback_prediction.training.calibration import apply_probability_calibrators


def predict_frame(
    candles: pd.DataFrame,
    artifact_path: str | Path,
    *,
    min_probability: float = 0.60,
    timeframe: str | None = None,
) -> pd.DataFrame:
    """Return calibrated class scores and BUY/SELL/WAIT signals for each candle.

    Artifacts with a validation-fitted calibrator return calibrated probabilities.
    Older artifacts return raw model scores. The abstention threshold is not a
    profitability claim.
    """
    if not 0.0 <= min_probability <= 1.0:
        raise ValueError("min_probability must be between 0 and 1.")

    if timeframe is not None:
        if timeframe not in {"M1", "M5"}:
            raise ValueError("timeframe must be M1 or M5.")
        if not isinstance(candles.index, pd.DatetimeIndex) or candles.index.tz is None:
            raise ValueError("Candle index must contain timezone-aware timestamps.")
        expected_interval = pd.Timedelta(
            60 if timeframe == "M1" else 300, unit="s"
        )
        median_interval = candles.index.to_series().diff().dropna().median()
        if median_interval != expected_interval:
            raise ValueError("Candle spacing does not match the requested timeframe.")

    artifact: dict[str, Any] = joblib.load(artifact_path)
    model = artifact.get("model")
    feature_columns = artifact.get("feature_columns")
    artifact_timeframe = artifact.get("timeframe")
    if timeframe is not None:
        if timeframe not in {"M1", "M5"}:
            raise ValueError("timeframe must be M1 or M5.")
        if artifact_timeframe != timeframe:
            raise ValueError(
                "Requested timeframe does not match model artifact; retrain or select the matching model."
            )
        if not isinstance(candles.index, pd.DatetimeIndex):
            raise ValueError("Candle data must use a DatetimeIndex.")
        seconds = 60 if timeframe == "M1" else 300
        intervals = candles.index.to_series().diff().dt.total_seconds()
        intervals = intervals[(intervals > 0) & (intervals <= seconds * 15)]
        if intervals.empty or not np.isclose(intervals.median(), seconds, atol=1):
            raise ValueError(
                f"Candle timestamps do not match the requested {timeframe} timeframe."
            )
    if model is None or not feature_columns:
        raise ValueError("Model artifact must contain model and feature_columns.")

    features = build_features(candles)
    missing = set(feature_columns).difference(features.columns)
    if missing:
        raise ValueError(f"Missing model features: {', '.join(sorted(missing))}")
    matrix = features.loc[:, feature_columns]
    probabilities = np.asarray(model.predict_proba(matrix), dtype=float)
    classes = np.asarray(model.classes_, dtype=int)
    if probabilities.shape != (len(candles), len(classes)):
        raise ValueError("Model probability output has an unexpected shape.")
    calibrators = artifact.get("calibrators")
    probabilities_calibrated = calibrators is not None
    if probabilities_calibrated:
        calibration_classes = np.asarray(artifact.get("calibration_classes"), dtype=int)
        if not np.array_equal(calibration_classes, classes):
            raise ValueError("Calibration classes do not match model classes.")
        probabilities = apply_probability_calibrators(probabilities, calibrators)

    best_indices = probabilities.argmax(axis=1)
    predicted_class = classes[best_indices]
    confidence = probabilities.max(axis=1)
    signal = np.where(
        (confidence >= min_probability) & (predicted_class != 0),
        predicted_class,
        0,
    ).astype("int8")

    result = pd.DataFrame(
        {
            "signal": signal,
            "predicted_class": predicted_class.astype("int8"),
            "confidence": confidence,
            "probabilities_calibrated": probabilities_calibrated,
            "prob_sell": 0.0,
            "prob_wait": 0.0,
            "prob_buy": 0.0,
            "atr_14": features["atr_14"],
        },
        index=candles.index,
    )
    probability_columns = {-1: "prob_sell", 0: "prob_wait", 1: "prob_buy"}
    for position, class_value in enumerate(classes):
        column = probability_columns.get(int(class_value))
        if column is not None:
            result[column] = probabilities[:, position]
    return result
