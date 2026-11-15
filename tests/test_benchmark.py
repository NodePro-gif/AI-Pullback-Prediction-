import numpy as np
import pandas as pd

from pullback_prediction.training.benchmark import benchmark_tabular_models


def test_benchmark_runs_hist_model_on_shared_time_folds():
    index = pd.date_range("2020-01-01", periods=120, freq="h", tz="UTC")
    rng = np.random.default_rng(4)
    x = rng.normal(size=120)
    features = pd.DataFrame({"signal": x, "noise": rng.normal(size=120)}, index=index)
    labels = pd.Series(np.where(x > 0, 1, -1), index=index)

    result = benchmark_tabular_models(
        features, labels, n_splits=3, gap_bars=5, max_iter=10, include_xgboost=False
    )

    assert set(result["summary"]) == {"hist_gradient_boosting"}
    assert len(result["folds"]["hist_gradient_boosting"]) == 3
    assert 0 <= result["summary"]["hist_gradient_boosting"]["weighted_accuracy"] <= 1


def test_benchmark_rejects_unaligned_data():
    index = pd.date_range("2020-01-01", periods=30, freq="h", tz="UTC")
    features = pd.DataFrame({"x": np.arange(30)}, index=index)
    labels = pd.Series(np.where(np.arange(30) % 2, 1, -1), index=index[::-1])
    try:
        benchmark_tabular_models(features, labels, include_xgboost=False)
    except ValueError as exc:
        assert "identical indexes" in str(exc)
    else:
        raise AssertionError("Expected unaligned indexes to fail.")
