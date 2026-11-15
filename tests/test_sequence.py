import numpy as np
import pandas as pd
import pytest

from pullback_prediction.training.sequence import make_sequence_windows


def test_sequence_windows_use_only_trailing_rows_and_preserve_end_timestamps():
    index = pd.date_range("2020-01-01", periods=8, freq="h", tz="UTC")
    features = pd.DataFrame({"x": np.arange(8), "y": np.arange(8) * 2}, index=index)
    labels = pd.Series(np.arange(8) % 3 - 1, index=index)

    windows = make_sequence_windows(features, labels, sequence_length=3)

    assert windows.features.shape == (6, 3, 2)
    assert windows.labels.tolist() == labels.iloc[2:].tolist()
    assert windows.timestamps.equals(index[2:])
    np.testing.assert_array_equal(windows.features[0, :, 0], [0, 1, 2])


def test_sequence_windows_drop_any_window_with_invalid_features_or_label():
    index = pd.date_range("2020-01-01", periods=8, freq="h", tz="UTC")
    features = pd.DataFrame({"x": np.arange(8, dtype=float)}, index=index)
    labels = pd.Series(np.ones(8), index=index)
    features.iloc[3, 0] = np.inf
    labels.iloc[6] = np.nan

    windows = make_sequence_windows(features, labels, sequence_length=3)

    assert index[3] not in windows.timestamps
    assert index[4] not in windows.timestamps
    assert index[6] not in windows.timestamps


def test_sequence_windows_reject_misaligned_indexes():
    index = pd.date_range("2020-01-01", periods=8, freq="h", tz="UTC")
    features = pd.DataFrame({"x": np.arange(8)}, index=index)
    labels = pd.Series(np.ones(8), index=index[::-1])
    with pytest.raises(ValueError, match="identical indexes"):
        make_sequence_windows(features, labels)

def test_sequence_model_architectures_forward_pass_when_torch_is_available():
    torch = pytest.importorskip("torch")
    from pullback_prediction.training.sequence import build_sequence_model

    values = torch.randn(4, 16, 6)
    for architecture in ("lstm", "transformer"):
        model = build_sequence_model(
            architecture,
            n_features=6,
            n_classes=3,
            hidden_dim=16,
            num_layers=1,
            max_sequence_length=16,
        )
        assert model(values).shape == (4, 3)


def test_sequence_training_runs_batched_validation_when_torch_is_available():
    pytest.importorskip("torch")
    from pullback_prediction.training.sequence import (
        build_sequence_model,
        train_sequence_model,
    )

    rng = np.random.default_rng(7)
    train_x = rng.normal(size=(24, 8, 4)).astype(np.float32)
    train_y = np.tile(np.array([-1, 0, 1]), 8)
    validation_x = rng.normal(size=(11, 8, 4)).astype(np.float32)
    validation_y = np.tile(np.array([-1, 0, 1]), 4)[:11]
    model = build_sequence_model(
        "lstm",
        n_features=4,
        n_classes=3,
        hidden_dim=8,
        num_layers=1,
    )

    result = train_sequence_model(
        model,
        train_x,
        train_y,
        validation_x,
        validation_y,
        epochs=2,
        batch_size=4,
        patience=2,
        device="cpu",
    )

    assert result["device"] == "cpu"
    assert result["mixed_precision"] is False
    assert result["classes"] == [-1, 0, 1]
    assert result["epochs_completed"] >= 1
    assert np.isfinite(result["best_validation_loss"])
