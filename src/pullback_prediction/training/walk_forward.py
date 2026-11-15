"""Expanding-window walk-forward evaluation with a candle-based leakage gap."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit


def walk_forward_evaluate(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    n_splits: int = 5,
    gap_bars: int = 12,
    max_iter: int = 200,
    learning_rate: float = 0.08,
    max_leaf_nodes: int = 31,
    l2_regularization: float = 1.0,
) -> dict[str, Any]:
    """Evaluate expanding training windows; gap is measured before invalid rows are removed."""
    if not features.index.equals(labels.index):
        raise ValueError("Features and labels must have identical indexes.")
    if not features.index.is_monotonic_increasing or not features.index.is_unique:
        raise ValueError("Input index must be unique and chronological.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if gap_bars < 0:
        raise ValueError("gap_bars cannot be negative.")
    if features.empty:
        raise ValueError("Cannot evaluate an empty dataset.")

    numeric = features.apply(pd.to_numeric, errors="coerce")
    finite_features = pd.Series(
        np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1),
        index=features.index,
    )
    valid = finite_features & labels.notna()
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap_bars)
    fold_results: list[dict[str, Any]] = []

    for fold_number, (train_positions, test_positions) in enumerate(
        splitter.split(features), start=1
    ):
        train_index = features.index[train_positions]
        test_index = features.index[test_positions]
        train_mask = valid.loc[train_index]
        test_mask = valid.loc[test_index]
        train_index = train_index[train_mask.to_numpy()]
        test_index = test_index[test_mask.to_numpy()]
        if len(train_index) == 0 or len(test_index) == 0:
            raise ValueError(f"Fold {fold_number} has no usable train or test rows.")

        train_x = numeric.loc[train_index]
        test_x = numeric.loc[test_index]
        train_y = labels.loc[train_index].astype(int)
        test_y = labels.loc[test_index].astype(int)
        if train_y.nunique() < 2:
            raise ValueError(f"Fold {fold_number} training labels contain fewer than two classes.")

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
        predictions = model.predict(test_x)
        probabilities = model.predict_proba(test_x)
        classes = set(model.classes_.tolist())
        test_classes = set(test_y.unique().tolist())
        result: dict[str, Any] = {
            "fold": fold_number,
            "train_rows": int(len(train_y)),
            "test_rows": int(len(test_y)),
            "train_start": train_index.min().isoformat(),
            "train_end": train_index.max().isoformat(),
            "test_start": test_index.min().isoformat(),
            "test_end": test_index.max().isoformat(),
            "accuracy": float(accuracy_score(test_y, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(test_y, predictions)),
            "log_loss": None,
        }
        if test_classes.issubset(classes):
            result["log_loss"] = float(
                log_loss(test_y, probabilities, labels=model.classes_)
            )
        else:
            result["log_loss_note"] = "Skipped: test fold contains a class absent from training."
        fold_results.append(result)

    total_rows = sum(fold["test_rows"] for fold in fold_results)
    aggregate = {
        "folds": len(fold_results),
        "test_rows": total_rows,
        "weighted_accuracy": sum(
            fold["accuracy"] * fold["test_rows"] for fold in fold_results
        )
        / total_rows,
        "mean_balanced_accuracy": float(
            np.mean([fold["balanced_accuracy"] for fold in fold_results])
        ),
        "mean_log_loss": (
            float(np.mean([fold["log_loss"] for fold in fold_results if fold["log_loss"] is not None]))
            if any(fold["log_loss"] is not None for fold in fold_results)
            else None
        ),
    }
    return {"summary": aggregate, "folds": fold_results}
