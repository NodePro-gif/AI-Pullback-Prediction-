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



def _run_fake_xgboost_benchmark(monkeypatch, labels):
    import sys
    import types

    instances = []

    class FakeXGBClassifier:
        def __init__(self, **parameters):
            self.parameters = parameters
            instances.append(self)

        def fit(self, features, targets):
            self.classes_ = np.unique(targets)
            return self

        def predict(self, features):
            return np.zeros(len(features), dtype=int)

        def predict_proba(self, features):
            return np.full(
                (len(features), len(self.classes_)),
                1.0 / len(self.classes_),
            )

    fake_module = types.ModuleType("xgboost")
    fake_module.XGBClassifier = FakeXGBClassifier
    monkeypatch.setitem(sys.modules, "xgboost", fake_module)
    index = pd.date_range("2020-01-01", periods=len(labels), freq="h", tz="UTC")
    rng = np.random.default_rng(21)
    features = pd.DataFrame(
        {"signal": rng.normal(size=len(index)), "noise": rng.normal(size=len(index))},
        index=index,
    )
    result = benchmark_tabular_models(
        features,
        pd.Series(labels, index=index),
        n_splits=2,
        gap_bars=2,
        max_iter=5,
        include_xgboost=True,
        xgb_estimators=5,
    )
    return result, instances


def test_xgboost_benchmark_uses_binary_objective_for_two_classes(monkeypatch):
    labels = np.tile(np.array([-1, 1]), 40)
    result, instances = _run_fake_xgboost_benchmark(monkeypatch, labels)

    assert "xgboost" in result["summary"]
    assert len(instances) == 2
    assert all(model.parameters["objective"] == "binary:logistic" for model in instances)
    assert all(model.parameters["eval_metric"] == "logloss" for model in instances)
    assert all("num_class" not in model.parameters for model in instances)


def test_xgboost_benchmark_uses_multiclass_objective_for_three_classes(monkeypatch):
    labels = np.tile(np.array([-1, 0, 1]), 30)
    result, instances = _run_fake_xgboost_benchmark(monkeypatch, labels)

    assert "xgboost" in result["summary"]
    assert len(instances) == 2
    assert all(model.parameters["objective"] == "multi:softprob" for model in instances)
    assert all(model.parameters["num_class"] == 3 for model in instances)
    assert all(model.parameters["eval_metric"] == "mlogloss" for model in instances)
