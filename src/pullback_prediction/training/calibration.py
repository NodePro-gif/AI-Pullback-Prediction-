"""One-vs-rest isotonic calibration for multiclass model probabilities."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_probability_calibrators(
    probabilities: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
) -> list[IsotonicRegression | None]:
    """Fit one isotonic map per class using a separate chronological period."""
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels)
    classes = np.asarray(classes)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
        raise ValueError("Probability columns must match the supplied classes.")
    if len(labels) != len(probabilities):
        raise ValueError("Probability and label row counts must match.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite.")

    calibrators: list[IsotonicRegression | None] = []
    for column, class_value in enumerate(classes):
        binary_labels = (labels == class_value).astype("int8")
        if np.unique(binary_labels).size < 2:
            calibrators.append(None)
            continue
        calibrator = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip",
        )
        calibrator.fit(probabilities[:, column], binary_labels)
        calibrators.append(calibrator)
    return calibrators


def apply_probability_calibrators(
    probabilities: np.ndarray,
    calibrators: list[IsotonicRegression | None],
) -> np.ndarray:
    """Apply one-vs-rest maps and renormalize rows to a probability simplex."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(calibrators):
        raise ValueError("Probability columns and calibrators must have the same number of columns.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite.")

    calibrated = np.empty_like(probabilities)
    for column, calibrator in enumerate(calibrators):
        if calibrator is None:
            calibrated[:, column] = probabilities[:, column]
        else:
            calibrated[:, column] = calibrator.predict(probabilities[:, column])

    calibrated = np.clip(calibrated, 0.0, 1.0)
    row_sums = calibrated.sum(axis=1, keepdims=True)
    zero_rows = row_sums[:, 0] <= 1e-12
    if zero_rows.any():
        calibrated[zero_rows] = probabilities[zero_rows]
        row_sums = calibrated.sum(axis=1, keepdims=True)
    zero_rows = row_sums[:, 0] <= 1e-12
    if zero_rows.any():
        calibrated[zero_rows] = 1.0 / probabilities.shape[1]
        row_sums = calibrated.sum(axis=1, keepdims=True)
    return calibrated / row_sums
