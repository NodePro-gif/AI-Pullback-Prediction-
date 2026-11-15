import numpy as np
import pytest

from pullback_prediction.training.calibration import (
    apply_probability_calibrators,
    fit_probability_calibrators,
)


def test_isotonic_calibration_returns_normalized_probabilities():
    rng = np.random.default_rng(17)
    probabilities = rng.dirichlet([2.0, 2.0, 2.0], size=300)
    labels = np.array([-1, 0, 1] * 100)
    classes = np.array([-1, 0, 1])

    calibrators = fit_probability_calibrators(probabilities, labels, classes)
    calibrated = apply_probability_calibrators(probabilities, calibrators)

    assert calibrated.shape == probabilities.shape
    assert np.isfinite(calibrated).all()
    assert (calibrated >= 0.0).all()
    assert (calibrated <= 1.0).all()
    assert np.allclose(calibrated.sum(axis=1), 1.0)


def test_calibration_falls_back_to_identity_for_unseen_class():
    probabilities = np.array(
        [[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.6, 0.3, 0.1], [0.3, 0.6, 0.1]]
    )
    labels = np.array([-1, 0, -1, 0])
    classes = np.array([-1, 0, 1])

    calibrators = fit_probability_calibrators(probabilities, labels, classes)

    assert calibrators[2] is None
    calibrated = apply_probability_calibrators(probabilities, calibrators)
    assert np.isfinite(calibrated).all()
    assert np.allclose(calibrated.sum(axis=1), 1.0)


def test_calibration_rejects_probability_class_shape_mismatch():
    with pytest.raises(ValueError, match="same number of columns"):
        apply_probability_calibrators(np.ones((2, 2)), [None, None, None])


@pytest.mark.parametrize(
    "probabilities",
    [
        np.array([[0.2, 0.2, 0.2]]),
        np.array([[1.1, -0.1, 0.0]]),
        np.array([[np.nan, 0.5, 0.5]]),
    ],
)
def test_calibration_rejects_invalid_probability_simplex(probabilities):
    with pytest.raises(ValueError):
        fit_probability_calibrators(
            probabilities,
            np.array([-1]),
            np.array([-1, 0, 1]),
        )


def test_calibration_rejects_labels_outside_supplied_classes():
    with pytest.raises(ValueError, match="supplied classes"):
        fit_probability_calibrators(
            np.array([[0.2, 0.3, 0.5]]),
            np.array([2]),
            np.array([-1, 0, 1]),
        )
