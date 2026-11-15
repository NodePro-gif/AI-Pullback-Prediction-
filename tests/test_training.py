import numpy as np
import pandas as pd
import pytest

from pullback_prediction.training.baseline import fit_baseline


def test_baseline_fits_only_matching_feature_schema_and_reports_metrics():
    rng = np.random.default_rng(7)
    train_x = pd.DataFrame(
        {"momentum": rng.normal(size=120), "volatility": rng.normal(size=120)}
    )
    train_y = pd.Series(np.where(train_x["momentum"] > 0, 1, -1))
    validation_x = pd.DataFrame(
        {"momentum": rng.normal(size=40), "volatility": rng.normal(size=40)}
    )
    validation_y = pd.Series(np.where(validation_x["momentum"] > 0, 1, -1))

    model, metrics = fit_baseline(
        train_x, train_y, validation_x, validation_y, max_iter=20
    )

    assert set(model.classes_) == {-1, 1}
    assert metrics["validation_rows"] == 40
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["log_loss"] is not None


def test_baseline_rejects_mismatched_feature_columns():
    train_x = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0]})
    validation_x = pd.DataFrame({"b": [0.0, 1.0]})

    with pytest.raises(ValueError, match="feature columns"):
        fit_baseline(
            train_x,
            pd.Series([-1, 1, -1, 1]),
            validation_x,
            pd.Series([-1, 1]),
            max_iter=5,
        )
