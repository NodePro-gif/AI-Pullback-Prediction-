import joblib
import numpy as np
import pandas as pd

from pullback_prediction.training import baseline


class FakeClassifier:
    classes_ = np.array([-1, 1])

    def predict(self, features):
        return np.where(features["feature"].to_numpy() > 0, 1, -1)

    def predict_proba(self, features):
        positive = 1.0 / (1.0 + np.exp(-features["feature"].to_numpy()))
        return np.column_stack((1.0 - positive, positive))


def test_training_pipeline_separates_train_calibration_evaluation_and_final_test(
    tmp_path, monkeypatch
):
    train_index = pd.date_range("2015-01-01", periods=8, freq="min", tz="UTC")
    calibration_index = pd.date_range("2021-01-01", periods=8, freq="min", tz="UTC")
    evaluation_index = pd.date_range("2023-01-01", periods=8, freq="min", tz="UTC")
    final_test_index = pd.date_range("2024-01-01", periods=8, freq="min", tz="UTC")
    index = train_index.append(calibration_index).append(evaluation_index).append(
        final_test_index
    )
    candles = pd.DataFrame({"placeholder": range(len(index))}, index=index)
    observed = {"dataset_indices": []}

    def fake_build_features(development):
        observed["development_index"] = development.index
        return pd.DataFrame(
            {
                "atr_14": np.ones(len(development)),
                "feature": np.linspace(-2.0, 2.0, len(development)),
            },
            index=development.index,
        )

    def fake_dataset(partition, features, horizon_bars, barrier_atr_multiple):
        observed["dataset_indices"].append(partition.index)
        labels = np.tile(np.array([-1, 1]), len(partition) // 2)
        return features, pd.Series(labels, index=partition.index)

    def fake_fit(train_x, train_y, validation_x, validation_y, **kwargs):
        observed["train_index"] = train_x.index
        observed["calibration_index"] = validation_x.index
        return FakeClassifier(), {}

    monkeypatch.setattr(baseline, "build_features", fake_build_features)
    monkeypatch.setattr(baseline, "_dataset", fake_dataset)
    monkeypatch.setattr(baseline, "fit_baseline", fake_fit)
    artifact_path = tmp_path / "baseline.joblib"

    result = baseline.train_baseline_from_frame(
        candles,
        artifact_path,
        horizon_bars=2,
        max_iter=5,
    )
    artifact = joblib.load(artifact_path)

    assert observed["train_index"].equals(train_index)
    assert observed["calibration_index"].equals(calibration_index)
    assert observed["dataset_indices"][2].equals(evaluation_index)
    assert observed["development_index"].max() < pd.Timestamp("2024-01-01", tz="UTC")
    assert artifact["training_period"] == "2015-01-01 through 2020-12-31"
    assert artifact["calibration_period"] == "2021-01-01 through 2022-12-31"
    assert artifact["evaluation_period"] == "2023-01-01 through 2023-12-31"
    assert result["calibration_rows"] == 8
    assert result["evaluation_rows"] == 8
