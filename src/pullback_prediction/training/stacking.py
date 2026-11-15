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
    ) -> "StackedPullbackClassifier":
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
