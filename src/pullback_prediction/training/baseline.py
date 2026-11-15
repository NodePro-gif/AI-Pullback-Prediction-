"""Leakage-aware gradient-boosting baseline training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss

from pullback_prediction.data.splits import chronological_split
from pullback_prediction.features.technical import build_features
from pullback_prediction.labels.barriers import make_barrier_labels


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
    """Fit on training data only and report untouched validation metrics."""
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
) -> dict[str, Any]:
    """Train on 2015–2020 and evaluate on 2021–2023; never load final test rows."""
    partitions = chronological_split(candles)
    train_candles = partitions["train"]
    validation_candles = partitions["validation"]
    if train_candles.empty or validation_candles.empty:
        raise ValueError("Train and validation partitions must both be non-empty.")

    development = pd.concat([train_candles, validation_candles])
    development_features = build_features(development)
    train_features = development_features.loc[train_candles.index]
    validation_features = development_features.loc[validation_candles.index]

    train_x, train_y = _dataset(
        train_candles, train_features, horizon_bars, barrier_atr_multiple
    )
    validation_x, validation_y = _dataset(
        validation_candles, validation_features, horizon_bars, barrier_atr_multiple
    )
    model, metrics = fit_baseline(
        train_x,
        train_y,
        validation_x,
        validation_y,
        max_iter=max_iter,
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "feature_columns": list(train_x.columns),
        "target_mapping": {-1: "SELL", 0: "WAIT", 1: "BUY"},
        "training_period": "2015-01-01 through 2020-12-31",
        "validation_period": "2021-01-01 through 2023-12-31",
        "horizon_bars": horizon_bars,
        "barrier_atr_multiple": barrier_atr_multiple,
        "metrics": metrics,
    }
    joblib.dump(artifact, destination)
    metrics_path = destination.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        **metrics,
        "train_rows": int(len(train_y)),
        "model_path": str(destination),
        "metrics_path": str(metrics_path),
    }
