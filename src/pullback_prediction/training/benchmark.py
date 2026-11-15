"""Comparable expanding-window benchmarks for tabular classifiers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit


def benchmark_tabular_models(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    n_splits: int = 5,
    gap_bars: int = 12,
    max_iter: int = 200,
    include_xgboost: bool = True,
    xgb_estimators: int = 300,
) -> dict[str, Any]:
    """Compare HistGradientBoosting and optional XGBoost on identical time folds."""
    if not features.index.equals(labels.index):
        raise ValueError("Features and labels must have identical indexes.")
    if not features.index.is_monotonic_increasing or not features.index.is_unique:
        raise ValueError("Input index must be unique and chronological.")
    if n_splits < 2 or gap_bars < 0:
        raise ValueError("n_splits must be >= 2 and gap_bars must be >= 0.")

    numeric = features.apply(pd.to_numeric, errors="coerce")
    valid = pd.Series(
        np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1), index=features.index
    ) & labels.notna()
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap_bars)
    model_factories: dict[str, Any] = {
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            class_weight="balanced",
            early_stopping=False,
            random_state=42,
        )
    }
    xgb_import_error = None
    if include_xgboost:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            xgb_import_error = str(exc)
        else:
            def make_xgboost(class_count: int) -> Any:
                parameters: dict[str, Any] = {
                    "n_estimators": xgb_estimators,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "subsample": 0.85,
                    "colsample_bytree": 0.85,
                    "reg_lambda": 2.0,
                    "tree_method": "hist",
                    "n_jobs": -1,
                    "random_state": 42,
                }
                if class_count == 2:
                    parameters.update(
                        objective="binary:logistic",
                        eval_metric="logloss",
                    )
                else:
                    parameters.update(
                        objective="multi:softprob",
                        num_class=class_count,
                        eval_metric="mlogloss",
                    )
                return XGBClassifier(**parameters)

            model_factories["xgboost"] = make_xgboost

    results: dict[str, list[dict[str, Any]]] = {
        name: [] for name in model_factories
    }
    for fold, (train_pos, test_pos) in enumerate(splitter.split(features), start=1):
        train_index = features.index[train_pos]
        test_index = features.index[test_pos]
        train_index = train_index[valid.loc[train_index].to_numpy()]
        test_index = test_index[valid.loc[test_index].to_numpy()]
        if not len(train_index) or not len(test_index):
            raise ValueError(f"Fold {fold} has no valid rows.")
        train_x, test_x = numeric.loc[train_index], numeric.loc[test_index]
        train_y, test_y = labels.loc[train_index].astype(int), labels.loc[test_index].astype(int)
        classes = np.sort(train_y.unique())
        if len(classes) < 2:
            raise ValueError(f"Fold {fold} training labels contain fewer than two classes.")
        class_map = {int(value): i for i, value in enumerate(classes)}
        reverse_map = np.asarray(classes, dtype=int)
        mapped_train = train_y.map(class_map)
        mapped_test = test_y.map(class_map)

        for name, factory in model_factories.items():
            model = factory(len(classes)) if name == "xgboost" else factory()
            if name == "xgboost":
                model.fit(train_x, mapped_train)
                predicted_indices = model.predict(test_x).astype(int)
                probabilities = model.predict_proba(test_x)
                predictions = reverse_map[predicted_indices]
            else:
                model.fit(train_x, train_y)
                predictions = model.predict(test_x)
                probabilities = model.predict_proba(test_x)
            row: dict[str, Any] = {
                "fold": fold,
                "train_rows": int(len(train_y)),
                "test_rows": int(len(test_y)),
                "train_end": train_index.max().isoformat(),
                "test_start": test_index.min().isoformat(),
                "test_end": test_index.max().isoformat(),
                "accuracy": float(accuracy_score(test_y, predictions)),
                "balanced_accuracy": float(balanced_accuracy_score(test_y, predictions)),
                "log_loss": None,
            }
            if mapped_test.notna().all():
                row["log_loss"] = float(
                    log_loss(mapped_test, probabilities, labels=np.arange(len(classes)))
                )
            results[name].append(row)

    summaries: dict[str, Any] = {}
    for name, folds in results.items():
        count = sum(item["test_rows"] for item in folds)
        summaries[name] = {
            "folds": len(folds),
            "test_rows": count,
            "weighted_accuracy": sum(
                item["accuracy"] * item["test_rows"] for item in folds
            ) / count,
            "mean_balanced_accuracy": float(
                np.mean([item["balanced_accuracy"] for item in folds])
            ),
            "mean_log_loss": (
                float(np.mean([item["log_loss"] for item in folds if item["log_loss"] is not None]))
                if any(item["log_loss"] is not None for item in folds)
                else None
            ),
        }
    return {
        "summary": summaries,
        "folds": results,
        "skipped": (
            {"xgboost": xgb_import_error}
            if include_xgboost and xgb_import_error is not None
            else {}
        ),
    }
