"""Train-only market regime clustering on causal volatility and trend features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DEFAULT_REGIME_FEATURES = (
    "atr_pct",
    "volatility_5",
    "volatility_60",
    "trend_strength_20_50",
    "volatility_ratio_5_60",
    "spread_to_atr",
)


@dataclass
class RegimeDetector:
    """A fitted scaler and KMeans model; fit only on the training period."""

    feature_columns: tuple[str, ...]
    scaler: StandardScaler
    model: KMeans

    @classmethod
    def fit(
        cls,
        features: pd.DataFrame,
        *,
        n_regimes: int = 4,
        feature_columns: tuple[str, ...] = DEFAULT_REGIME_FEATURES,
        random_state: int = 42,
    ) -> "RegimeDetector":
        """Fit clusters using finite rows from the supplied training features."""
        if n_regimes < 2:
            raise ValueError("n_regimes must be at least 2.")
        if not features.index.is_monotonic_increasing or not features.index.is_unique:
            raise ValueError("Feature index must be unique and chronological.")
        missing = set(feature_columns).difference(features.columns)
        if missing:
            raise ValueError(f"Missing regime features: {', '.join(sorted(missing))}.")
        values = features.loc[:, feature_columns].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
        valid = np.isfinite(values).all(axis=1)
        training_values = values[valid]
        if len(training_values) < n_regimes:
            raise ValueError("Not enough finite training rows for the requested regimes.")
        if np.unique(training_values, axis=0).shape[0] < n_regimes:
            raise ValueError("Training features contain fewer unique rows than regimes.")

        scaler = StandardScaler()
        scaled = scaler.fit_transform(training_values)
        model = KMeans(
            n_clusters=n_regimes,
            n_init=10,
            random_state=random_state,
        )
        model.fit(scaled)
        return cls(tuple(feature_columns), scaler, model)

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Assign regimes; invalid warm-up rows remain NaN."""
        missing = set(self.feature_columns).difference(features.columns)
        if missing:
            raise ValueError(f"Missing regime features: {', '.join(sorted(missing))}.")
        values = features.loc[:, self.feature_columns].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
        valid = np.isfinite(values).all(axis=1)
        regime = np.full(len(features), np.nan, dtype=float)
        distance = np.full(len(features), np.nan, dtype=float)
        if valid.any():
            scaled = self.scaler.transform(values[valid])
            distances = self.model.transform(scaled)
            regime[valid] = distances.argmin(axis=1)
            distance[valid] = distances.min(axis=1)
        return pd.DataFrame(
            {"regime_id": regime, "regime_distance": distance},
            index=features.index,
        )
