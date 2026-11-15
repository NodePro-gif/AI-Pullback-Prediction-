"""Leakage-aware gradient-boosting baseline training and calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss

from pullback_prediction.data.splits import chronological_split
from pullback_prediction.features.technical import build_features
from pullback_prediction.labels.barriers import make_barrier_labels
from pullback_prediction.training.calibration import (
    apply_probability_calibrators,
    fit_probability_calibrators,
)

CALIBRATION_EVALUATION_CUTOFF = pd.Timestamp("2023-01-01", tz="UTC")


def _dataset(
    candles: pd.DataFrame,
    features: pd.DataFrame,
    horizon_bars: int,
    barrier_atr_multiple: float,
) -> tuple[pd.DataFrame, pd.Series]:
    labels = make_barrier_labels(
        candles,
        features["atr_14"],
        horizon_bars=horizon_bars,
        barrier_atr_multiple=barrier_atr_multiple,
    )
    valid = labels.notna() & features.notna().all(axis=1)
    return features.loc[valid], labels.loc[valid].astype("int8")


def fit_baseline(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    validation_x: pd.DataFrame,
    validation_y: pd.Series,
    *,
    max_iter: int = 200,
    learning_rate: float = 0.08,
    max_leaf_nodes: int = 31,
    l2_regularization: float = 1.0,
) -> tuple[HistGradientBoostingClassifier, dict[str, Any]]:
    """Fit on training data only and report metrics on a separate holdout."""
    if list(train_x.columns) != list(validation_x.columns):
        raise ValueError("Training and validation feature columns must match.")
    if len(train_x) != len(train_y) or len(validation_x) != len(validation_y):
        raise ValueError("Feature and target row counts must match.")
    if train_y.nunique() < 2:
        raise ValueError("Training target must contain at least two classes.")
    if validation_x.empty:
        raise ValueError("Validation partition has no usable rows.")

    model = HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        l2_regularization=l2_regularization,
        class_weight="balanced",
        early_stopping=False,
        random_state=42,
    )
    model.fit(train_x, train_y)
    predictions = model.predict(validation_x)
    probabilities = model.predict_proba(validation_x)
    known_classes = set(model.classes_.tolist())
    validation_classes = set(validation_y.unique().tolist())

    metrics: dict[str, Any] = {
        "validation_rows": int(len(validation_y)),
        "accuracy": float(accuracy_score(validation_y, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(validation_y, predictions)
        ),
        "classes": [int(value) for value in model.classes_],
        "validation_classes": [int(value) for value in sorted(validation_classes)],
        "log_loss": None,
    }
    if validation_classes.issubset(known_classes):
        metrics["log_loss"] = float(
            log_loss(validation_y, probabilities, labels=model.classes_)
        )
    else:
        metrics["log_loss_note"] = (
            "Omitted because validation contains a class absent from training."
        )
    return model, metrics


def train_baseline_from_frame(
    candles: pd.DataFrame,
    output_path: str | Path,
    *,
    horizon_bars: int = 12,
    barrier_atr_multiple: float = 1.5,
    max_iter: int = 200,
    timeframe: str | None = None,
) -> dict[str, Any]:
    """Train 2015–2020, calibrate 2021–2022, evaluate 2023; exclude final test."""
    if timeframe not in {None, "M1", "M5"}:
        raise ValueError("timeframe must be M1 or M5.")
    if timeframe is not None:
        if not isinstance(candles.index, pd.DatetimeIndex) or candles.index.tz is None:
            raise ValueError("Candle index must contain timezone-aware timestamps.")
        expected_interval = pd.Timedelta(
            60 if timeframe == "M1" else 300, unit="s"
        )
        median_interval = candles.index.to_series().diff().dropna().median()
        if median_interval != expected_interval:
            raise ValueError("Candle spacing does not match the requested timeframe.")
    partitions = chronological_split(candles)
    train_candles = partitions["train"]
    validation_candles = partitions["validation"]
    if train_candles.empty or validation_candles.empty:
        raise ValueError("Train and validation partitions must both be non-empty.")

    calibration_candles = validation_candles.loc[
        validation_candles.index < CALIBRATION_EVALUATION_CUTOFF
    ]
    evaluation_candles = validation_candles.loc[
        validation_candles.index >= CALIBRATION_EVALUATION_CUTOFF
    ]
    if calibration_candles.empty or evaluation_candles.empty:
        raise ValueError(
            "Validation must contain both 2021–2022 calibration and 2023 evaluation rows."
        )

    development = pd.concat([train_candles, validation_candles])
    development_features = build_features(development)
    train_features = development_features.loc[train_candles.index]
    calibration_features = development_features.loc[calibration_candles.index]
    evaluation_features = development_features.loc[evaluation_candles.index]

    train_x, train_y = _dataset(
        train_candles, train_features, horizon_bars, barrier_atr_multiple
    )
    calibration_x, calibration_y = _dataset(
        calibration_candles,
        calibration_features,
        horizon_bars,
        barrier_atr_multiple,
    )
    evaluation_x, evaluation_y = _dataset(
        evaluation_candles,
        evaluation_features,
        horizon_bars,
        barrier_atr_multiple,
    )
    if calibration_x.empty or evaluation_x.empty:
        raise ValueError("Calibration and evaluation partitions need usable labeled rows.")

    model, _ = fit_baseline(
        train_x,
        train_y,
        calibration_x,
        calibration_y,
        max_iter=max_iter,
    )
    classes = np.asarray(model.classes_, dtype=int)
    calibration_raw = model.predict_proba(calibration_x)
    calibrators = fit_probability_calibrators(
        calibration_raw,
        calibration_y.to_numpy(),
        classes,
    )
    evaluation_raw = model.predict_proba(evaluation_x)
    evaluation_probabilities = apply_probability_calibrators(
        evaluation_raw,
        calibrators,
    )
    predictions = classes[evaluation_probabilities.argmax(axis=1)]
    evaluation_classes = set(evaluation_y.unique().tolist())
    known_classes = set(classes.tolist())

    one_hot = np.column_stack(
        [(evaluation_y.to_numpy() == class_value) for class_value in classes]
    ).astype(float)
    metrics: dict[str, Any] = {
        "calibration_rows": int(len(calibration_y)),
        "evaluation_rows": int(len(evaluation_y)),
        "accuracy": float(accuracy_score(evaluation_y, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(evaluation_y, predictions)
        ),
        "multiclass_brier_score": float(
            np.mean(np.sum((one_hot - evaluation_probabilities) ** 2, axis=1))
        ),
        "log_loss": None,
        "classes": [int(value) for value in classes],
        "evaluation_classes": [
            int(value) for value in sorted(evaluation_classes)
        ],
    }
    if evaluation_classes.issubset(known_classes):
        metrics["log_loss"] = float(
            log_loss(evaluation_y, evaluation_probabilities, labels=classes)
        )
    else:
        metrics["log_loss_note"] = (
            "Omitted because evaluation contains a class absent from training."
        )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "feature_columns": list(train_x.columns),
        "timeframe": timeframe,
        "calibrators": calibrators,
        "calibration_classes": [int(value) for value in classes],
        "target_mapping": {-1: "SELL", 0: "WAIT", 1: "BUY"},
        "training_period": "2015-01-01 through 2020-12-31",
        "calibration_period": "2021-01-01 through 2022-12-31",
        "evaluation_period": "2023-01-01 through 2023-12-31",
        "final_test_period": "2024-01-01 onward; not accessed",
        "horizon_bars": horizon_bars,
        "barrier_atr_multiple": barrier_atr_multiple,
        "metrics": metrics,
    }
    joblib.dump(artifact, destination)
    metrics_path = destination.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        **metrics,
        "train_rows": int(len(train_y)),
        "model_path": str(destination),
        "metrics_path": str(metrics_path),
    }
