"""Leakage-aware probability stacking with expanding-window out-of-fold predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class StackedPullbackClassifier:
    """A tabular base model and a logistic meta-model trained on OOF scores."""

    base_model: HistGradientBoostingClassifier
    meta_model: LogisticRegression
    classes_: np.ndarray
    feature_columns: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        features: pd.DataFrame,
        labels: pd.Series,
        *,
        n_splits: int = 5,
        gap_bars: int = 12,
        max_iter: int = 200,
        random_state: int = 42,
    ) -> StackedPullbackClassifier:
        """Generate chronological OOF scores, fit meta-model, then refit base model."""
        if not features.index.equals(labels.index):
            raise ValueError("Features and labels must have identical indexes.")
        if (
            not isinstance(features.index, pd.DatetimeIndex)
            or not features.index.is_monotonic_increasing
            or not features.index.is_unique
        ):
            raise ValueError("Features require a unique chronological DatetimeIndex.")
        if n_splits < 2 or gap_bars < 0 or max_iter < 1:
            raise ValueError("Invalid stacking hyperparameters.")
        numeric = features.apply(pd.to_numeric, errors="coerce")
        values = numeric.to_numpy(dtype=float)
        valid = np.isfinite(values).all(axis=1) & labels.notna().to_numpy()
        classes = np.sort(labels.loc[valid].astype(int).unique())
        if len(classes) < 2:
            raise ValueError("Stacking requires at least two target classes.")

        splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap_bars)
        oof = np.full((len(features), len(classes)), np.nan, dtype=float)
        for fold, (train_pos, test_pos) in enumerate(splitter.split(features), start=1):
            train_pos = train_pos[valid[train_pos]]
            test_pos = test_pos[valid[test_pos]]
            if not len(train_pos) or not len(test_pos):
                continue
            train_y = labels.iloc[train_pos].astype(int)
            if train_y.nunique() < 2:
                continue
            model = HistGradientBoostingClassifier(
                max_iter=max_iter,
                learning_rate=0.08,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                class_weight="balanced",
                early_stopping=False,
                random_state=random_state + fold,
            )
            model.fit(numeric.iloc[train_pos], train_y)
            fold_probabilities = model.predict_proba(numeric.iloc[test_pos])
            for column, class_value in enumerate(model.classes_):
                destination = int(np.searchsorted(classes, class_value))
                oof[test_pos, destination] = fold_probabilities[:, column]

        meta_valid = np.isfinite(oof).all(axis=1) & valid
        meta_labels = labels.iloc[np.flatnonzero(meta_valid)].astype(int)
        if meta_labels.nunique() != len(classes):
            raise ValueError(
                "OOF rows must contain every target class; increase data or reduce n_splits."
            )
        meta_model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        )
        meta_model.fit(oof[meta_valid], meta_labels)

        train_x = numeric.loc[valid]
        train_y = labels.loc[valid].astype(int)
        base_model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            class_weight="balanced",
            early_stopping=False,
            random_state=random_state,
        )
        base_model.fit(train_x, train_y)
        return cls(
            base_model=base_model,
            meta_model=meta_model,
            classes_=classes,
            feature_columns=tuple(features.columns),
        )

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return stacked probabilities ordered by the fitted target classes."""
        if tuple(features.columns) != self.feature_columns:
            raise ValueError("Prediction feature columns must match training columns.")
        numeric = features.apply(pd.to_numeric, errors="coerce")
        base_probabilities = self.base_model.predict_proba(numeric)
        aligned = np.zeros((len(features), len(self.classes_)), dtype=float)
        for column, class_value in enumerate(self.base_model.classes_):
            destination = int(np.searchsorted(self.classes_, class_value))
            aligned[:, destination] = base_probabilities[:, column]
        meta_probabilities = self.meta_model.predict_proba(aligned)
        output = np.zeros_like(aligned)
        for column, class_value in enumerate(self.meta_model.classes_):
            destination = int(np.searchsorted(self.classes_, class_value))
            output[:, destination] = meta_probabilities[:, column]
        return output


def train_stacked_from_frame(
    candles: pd.DataFrame,
    output_path: str,
    *,
    timeframe: str = "M1",
    horizon_bars: int = 12,
    barrier_atr_multiple: float = 1.5,
    n_splits: int = 5,
    gap_bars: int | None = None,
    max_iter: int = 200,
) -> dict[str, Any]:
    """Train OOF stack, calibrate on 2021–2022, evaluate on 2023, exclude 2024+."""
    import json
    from pathlib import Path

    import joblib
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss

    from pullback_prediction.data.splits import chronological_split
    from pullback_prediction.features.technical import build_features
    from pullback_prediction.labels.barriers import make_barrier_labels
    from pullback_prediction.training.calibration import (
        apply_probability_calibrators,
        fit_probability_calibrators,
    )

    if timeframe not in {"M1", "M5"}:
        raise ValueError("timeframe must be M1 or M5.")
    effective_gap = horizon_bars if gap_bars is None else gap_bars
    if effective_gap < horizon_bars:
        raise ValueError("gap_bars must be at least horizon_bars.")
    partitions = chronological_split(candles)
    training = partitions["train"]
    validation = partitions["validation"]
    calibration_cutoff = pd.Timestamp("2023-01-01", tz="UTC")
    calibration = validation.loc[validation.index < calibration_cutoff]
    evaluation = validation.loc[validation.index >= calibration_cutoff]
    if training.empty or calibration.empty or evaluation.empty:
        raise ValueError("Train, 2021–2022 calibration and 2023 evaluation are required.")

    development = pd.concat([training, validation])
    features = build_features(development)
    train_x = features.loc[training.index]
    train_y = make_barrier_labels(
        training,
        train_x["atr_14"],
        horizon_bars=horizon_bars,
        barrier_atr_multiple=barrier_atr_multiple,
    )
    train_valid = train_x.notna().all(axis=1) & train_y.notna()
    calibration_x = features.loc[calibration.index]
    calibration_y = make_barrier_labels(
        calibration,
        calibration_x["atr_14"],
        horizon_bars=horizon_bars,
        barrier_atr_multiple=barrier_atr_multiple,
    )
    calibration_valid = calibration_x.notna().all(axis=1) & calibration_y.notna()
    evaluation_x = features.loc[evaluation.index]
    evaluation_y = make_barrier_labels(
        evaluation,
        evaluation_x["atr_14"],
        horizon_bars=horizon_bars,
        barrier_atr_multiple=barrier_atr_multiple,
    )
    evaluation_valid = evaluation_x.notna().all(axis=1) & evaluation_y.notna()
    train_x, train_y = train_x.loc[train_valid], train_y.loc[train_valid].astype(int)
    calibration_x = calibration_x.loc[calibration_valid]
    calibration_y = calibration_y.loc[calibration_valid].astype(int)
    evaluation_x = evaluation_x.loc[evaluation_valid]
    evaluation_y = evaluation_y.loc[evaluation_valid].astype(int)
    if calibration_x.empty or evaluation_x.empty:
        raise ValueError("Calibration and evaluation require valid labeled rows.")

    model = StackedPullbackClassifier.fit(
        train_x,
        train_y,
        n_splits=n_splits,
        gap_bars=effective_gap,
        max_iter=max_iter,
    )
    classes = model.classes_
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
    one_hot = np.column_stack(
        [evaluation_y.to_numpy() == value for value in classes]
    ).astype(float)
    evaluation_classes = set(evaluation_y.unique().tolist())
    known_classes = set(classes.tolist())
    metrics: dict[str, Any] = {
        "train_rows": int(len(train_y)),
        "calibration_rows": int(len(calibration_y)),
        "evaluation_rows": int(len(evaluation_y)),
        "accuracy": float(accuracy_score(evaluation_y, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(evaluation_y, predictions)
        ),
        "multiclass_brier_score": float(
            np.mean(np.sum((one_hot - evaluation_probabilities) ** 2, axis=1))
        ),
        "log_loss": (
            float(log_loss(evaluation_y, evaluation_probabilities, labels=classes))
            if evaluation_classes.issubset(known_classes)
            else None
        ),
        "classes": [int(value) for value in classes],
    }
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
    return {**metrics, "model_path": str(destination), "metrics_path": str(metrics_path)}
