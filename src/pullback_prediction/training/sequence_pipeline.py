"""Leakage-aware end-to-end LSTM/Transformer training pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

from pullback_prediction.data.splits import chronological_split
from pullback_prediction.features.technical import build_features
from pullback_prediction.labels.barriers import make_barrier_labels
from pullback_prediction.training.calibration import (
    apply_probability_calibrators,
    fit_probability_calibrators,
)
from pullback_prediction.training.sequence import (
    build_sequence_model,
    make_sequence_windows,
    train_sequence_model,
)


def _partition_sequences(
    candles: pd.DataFrame,
    features: pd.DataFrame,
    *,
    horizon_bars: int,
    barrier_atr_multiple: float,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = make_barrier_labels(
        candles,
        features["atr_14"],
        horizon_bars=horizon_bars,
        barrier_atr_multiple=barrier_atr_multiple,
    )
    windows = make_sequence_windows(features, labels, sequence_length=sequence_length)
    return windows.features, windows.labels


def train_sequence_from_frame(
    candles: pd.DataFrame,
    output_path: str | Path,
    *,
    architecture: str = "lstm",
    timeframe: str = "M1",
    sequence_length: int = 128,
    horizon_bars: int = 12,
    barrier_atr_multiple: float = 1.5,
    hidden_dim: int = 96,
    num_layers: int = 2,
    dropout: float = 0.15,
    epochs: int = 30,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    patience: int = 5,
    device: str | None = None,
    mixed_precision: bool = True,
    num_workers: int = 0,
) -> dict[str, Any]:
    """Fit 2015–2020; early-stop/calibrate 2021–2022; evaluate 2023 only."""
    if architecture not in {"lstm", "transformer"}:
        raise ValueError("architecture must be 'lstm' or 'transformer'.")
    if timeframe not in {"M1", "M5"}:
        raise ValueError("timeframe must be M1 or M5.")
    if sequence_length < 2 or horizon_bars < 1:
        raise ValueError("sequence_length and horizon_bars must be positive.")
    partitions = chronological_split(candles)
    training = partitions["train"]
    validation = partitions["validation"]
    early_stop_cutoff = pd.Timestamp("2022-01-01", tz="UTC")
    evaluation_cutoff = pd.Timestamp("2023-01-01", tz="UTC")
    early_stopping = validation.loc[validation.index < early_stop_cutoff]
    calibration = validation.loc[
        (validation.index >= early_stop_cutoff)
        & (validation.index < evaluation_cutoff)
    ]
    evaluation = validation.loc[validation.index >= evaluation_cutoff]
    if training.empty or early_stopping.empty or calibration.empty or evaluation.empty:
        raise ValueError(
            "2015–2020 train, 2021 early stopping, 2022 calibration and 2023 evaluation are required."
        )

    # Only development candles are passed to feature engineering. 2024+ is never read.
    development = pd.concat([training, validation])
    all_features = build_features(development)
    train_features = all_features.loc[training.index]
    early_stopping_features = all_features.loc[early_stopping.index]
    calibration_features = all_features.loc[calibration.index]
    evaluation_features = all_features.loc[evaluation.index]

    train_labels = make_barrier_labels(
        training, train_features["atr_14"], horizon_bars, barrier_atr_multiple
    )
    train_valid = train_features.notna().all(axis=1) & train_labels.notna()
    if not train_valid.any():
        raise ValueError("Training partition has no usable sequence labels.")
    scaler = StandardScaler()
    scaler.fit(train_features.loc[train_valid])
    columns = list(train_features.columns)

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            scaler.transform(frame.loc[:, columns]),
            index=frame.index,
            columns=columns,
        )

    train_x, train_y = _partition_sequences(
        training, transform(train_features), horizon_bars=horizon_bars,
        barrier_atr_multiple=barrier_atr_multiple, sequence_length=sequence_length,
    )
    early_stopping_x, early_stopping_y = _partition_sequences(
        early_stopping, transform(early_stopping_features), horizon_bars=horizon_bars,
        barrier_atr_multiple=barrier_atr_multiple, sequence_length=sequence_length,
    )
    calibration_x, calibration_y = _partition_sequences(
        calibration, transform(calibration_features), horizon_bars=horizon_bars,
        barrier_atr_multiple=barrier_atr_multiple, sequence_length=sequence_length,
    )
    evaluation_x, evaluation_y = _partition_sequences(
        evaluation, transform(evaluation_features), horizon_bars=horizon_bars,
        barrier_atr_multiple=barrier_atr_multiple, sequence_length=sequence_length,
    )
    if not len(early_stopping_y) or not len(calibration_y) or not len(evaluation_y):
        raise ValueError("Early stopping, calibration and evaluation need usable sequence windows.")
    classes = np.sort(np.unique(train_y))
    if len(classes) < 2:
        raise ValueError("Training sequence labels must contain at least two classes.")
    if not set(np.unique(early_stopping_y)).issubset(set(classes.tolist())):
        raise ValueError("Early-stopping labels contain a class absent from training.")
    if not set(np.unique(calibration_y)).issubset(set(classes.tolist())):
        raise ValueError("Calibration contains a class absent from training.")
    if not set(np.unique(evaluation_y)).issubset(set(classes.tolist())):
        raise ValueError("Evaluation contains a class absent from training.")

    model_config = {
        "architecture": architecture,
        "n_features": len(columns),
        "n_classes": len(classes),
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "max_sequence_length": sequence_length,
    }
    model = build_sequence_model(**model_config)
    trained = train_sequence_model(
        model, train_x, train_y, early_stopping_x, early_stopping_y,
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        patience=patience, device=device, mixed_precision=mixed_precision,
        num_workers=num_workers,
    )
    import torch

    def probabilities(values: np.ndarray) -> np.ndarray:
        trained["model"].eval()
        target_device = next(trained["model"].parameters()).device
        with torch.inference_mode():
            tensor = torch.as_tensor(values, dtype=torch.float32, device=target_device)
            logits = trained["model"](tensor)
            mapped = torch.softmax(logits.float(), dim=1).cpu().numpy()
        return mapped

    calibration_raw = probabilities(calibration_x)
    calibrators = fit_probability_calibrators(calibration_raw, calibration_y, classes)
    evaluation_probabilities = apply_probability_calibrators(
        probabilities(evaluation_x), calibrators
    )
    predictions = classes[evaluation_probabilities.argmax(axis=1)]
    one_hot = np.column_stack([evaluation_y == value for value in classes]).astype(float)
    metrics = {
        "train_windows": int(len(train_y)),
        "early_stopping_windows": int(len(early_stopping_y)),
        "calibration_windows": int(len(calibration_y)),
        "evaluation_windows": int(len(evaluation_y)),
        "accuracy": float(accuracy_score(evaluation_y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(evaluation_y, predictions)),
        "log_loss": float(log_loss(evaluation_y, evaluation_probabilities, labels=classes)),
        "multiclass_brier_score": float(
            np.mean(np.sum((one_hot - evaluation_probabilities) ** 2, axis=1))
        ),
        "best_validation_loss": float(trained["best_validation_loss"]),
        "epochs_completed": int(trained["epochs_completed"]),
        "device": trained["device"],
        "mixed_precision": trained["mixed_precision"],
        "classes": [int(value) for value in classes],
    }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    weights_path = destination.with_suffix(".pt")
    torch.save(
        {"state_dict": trained["model"].state_dict(), "model_config": model_config},
        weights_path,
    )
    joblib.dump(
        {
            "artifact_type": "sequence_classifier",
            "weights_path": weights_path.name,
            "model_config": model_config,
            "scaler": scaler,
            "feature_columns": columns,
            "timeframe": timeframe,
            "classes": [int(value) for value in classes],
            "calibrators": calibrators,
            "calibration_classes": [int(value) for value in classes],
            "sequence_length": sequence_length,
            "training_period": "2015-01-01 through 2020-12-31",
            "early_stopping_period": "2021-01-01 through 2021-12-31",
            "calibration_period": "2022-01-01 through 2022-12-31",
            "evaluation_period": "2023-01-01 through 2023-12-31",
            "final_test_period": "2024-01-01 onward; not accessed",
            "metrics": metrics,
        },
        destination,
    )
    destination.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        **metrics,
        "artifact_path": str(destination),
        "weights_path": str(weights_path),
    }
