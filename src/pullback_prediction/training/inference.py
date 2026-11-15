"""Model inference and confidence-based abstention."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from pullback_prediction.features.technical import build_features
from pullback_prediction.training.calibration import apply_probability_calibrators


def _predict_sequence_artifact(
    candles: pd.DataFrame,
    artifact: dict[str, Any],
    artifact_path: Path,
    min_probability: float,
) -> pd.DataFrame:
    """Load a saved PyTorch sequence checkpoint and align predictions to candles."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Install optional PyTorch dependencies: pip install -e '.[deep-learning]'"
        ) from exc

    from pullback_prediction.training.sequence import (
        build_sequence_model,
        make_sequence_windows,
    )

    columns = artifact.get("feature_columns")
    if not columns:
        raise ValueError("Sequence artifact has no feature_columns.")
    features = build_features(candles)
    missing = set(columns).difference(features.columns)
    if missing:
        raise ValueError(f"Missing model features: {', '.join(sorted(missing))}")
    scaled = pd.DataFrame(
        artifact["scaler"].transform(features.loc[:, columns]),
        index=features.index,
        columns=columns,
    )
    dummy_labels = pd.Series(0, index=scaled.index)
    windows = make_sequence_windows(
        scaled, dummy_labels, sequence_length=int(artifact["sequence_length"])
    )
    classes = np.asarray(artifact["classes"], dtype=int)
    result = pd.DataFrame(
        {
            "signal": np.zeros(len(candles), dtype="int8"),
            "predicted_class": np.zeros(len(candles), dtype="int8"),
            "confidence": np.zeros(len(candles), dtype=float),
            "probabilities_calibrated": bool(artifact.get("calibrators") is not None),
            "prob_sell": np.zeros(len(candles), dtype=float),
            "prob_wait": np.zeros(len(candles), dtype=float),
            "prob_buy": np.zeros(len(candles), dtype=float),
            "atr_14": features["atr_14"],
        },
        index=candles.index,
    )
    if not len(windows.features):
        return result
    model = build_sequence_model(**artifact["model_config"])
    weights_path = artifact_path.parent / artifact["weights_path"]
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.inference_mode():
        tensor = torch.as_tensor(windows.features, dtype=torch.float32)
        probabilities = torch.softmax(model(tensor).float(), dim=1).numpy()
    calibrators = artifact.get("calibrators")
    if calibrators is not None:
        calibration_classes = np.asarray(artifact["calibration_classes"], dtype=int)
        if not np.array_equal(calibration_classes, classes):
            raise ValueError("Calibration classes do not match sequence model classes.")
        probabilities = apply_probability_calibrators(probabilities, calibrators)
    best = probabilities.argmax(axis=1)
    predicted = classes[best]
    confidence = probabilities.max(axis=1)
    positions = candles.index.get_indexer(windows.timestamps)
    result.iloc[positions, result.columns.get_loc("predicted_class")] = predicted.astype("int8")
    result.iloc[positions, result.columns.get_loc("confidence")] = confidence
    result.iloc[positions, result.columns.get_loc("signal")] = np.where(
        (confidence >= min_probability) & (predicted != 0), predicted, 0
    ).astype("int8")
    probability_columns = {-1: "prob_sell", 0: "prob_wait", 1: "prob_buy"}
    for column, class_value in enumerate(classes):
        name = probability_columns.get(int(class_value))
        if name is not None:
            result.iloc[positions, result.columns.get_loc(name)] = probabilities[:, column]
    return result


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

    if timeframe is not None and timeframe not in {"M1", "M5"}:
        raise ValueError("timeframe must be M1 or M5.")

    artifact: dict[str, Any] = joblib.load(artifact_path)
    if artifact.get("artifact_type") == "sequence_classifier":
        return _predict_sequence_artifact(
            candles, artifact, Path(artifact_path), min_probability
        )
    model = artifact.get("model")
    feature_columns = artifact.get("feature_columns")
    artifact_timeframe = artifact.get("timeframe")
    if timeframe is not None:
        if timeframe not in {"M1", "M5"}:
            raise ValueError("timeframe must be M1 or M5.")
        if artifact_timeframe != timeframe:
            raise ValueError(
                "Requested timeframe does not match model artifact; "
                "retrain or select the matching model."
            )
        if not isinstance(candles.index, pd.DatetimeIndex) or candles.index.tz is None:
            raise ValueError("Candle index must contain timezone-aware timestamps.")
        seconds = 60 if timeframe == "M1" else 300
        intervals = candles.index.to_series().diff().dt.total_seconds()
        intervals = intervals[(intervals > 0) & (intervals <= seconds * 15)]
        if intervals.empty or not np.isclose(intervals.median(), seconds, atol=1):
            raise ValueError(
                f"Candle spacing does not match the requested {timeframe} timeframe."
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
