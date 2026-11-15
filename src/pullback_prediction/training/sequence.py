"""GPU-aware LSTM and Transformer sequence models (optional PyTorch dependency)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SequenceWindows:
    features: np.ndarray
    labels: np.ndarray
    timestamps: pd.Index


def make_sequence_windows(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    sequence_length: int = 128,
) -> SequenceWindows:
    """Build trailing-only windows; call separately for each chronological split."""
    if not features.index.equals(labels.index):
        raise ValueError("Features and labels must have identical indexes.")
    if not features.index.is_monotonic_increasing or not features.index.is_unique:
        raise ValueError("Input index must be unique and chronological.")
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2.")
    matrix = features.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    target = labels.to_numpy()
    windows: list[np.ndarray] = []
    targets: list[int] = []
    timestamps: list[Any] = []
    for end in range(sequence_length - 1, len(features)):
        start = end - sequence_length + 1
        window = matrix[start : end + 1]
        label = target[end]
        if not np.isfinite(window).all() or pd.isna(label):
            continue
        windows.append(window)
        targets.append(int(label))
        timestamps.append(features.index[end])
    if not windows:
        return SequenceWindows(
            np.empty((0, sequence_length, features.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            pd.Index([], dtype=features.index.dtype),
        )
    return SequenceWindows(
        np.stack(windows).astype(np.float32, copy=False),
        np.asarray(targets, dtype=np.int64),
        pd.Index(timestamps),
    )


def build_sequence_model(
    architecture: str,
    *,
    n_features: int,
    n_classes: int = 3,
    hidden_dim: int = 96,
    num_layers: int = 2,
    dropout: float = 0.15,
    max_sequence_length: int = 512,
):
    """Create an LSTM or Transformer classifier; PyTorch is imported on demand."""
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError("Install the optional ML dependencies: pip install -e '.[ml]'") from exc

    if architecture not in {"lstm", "transformer"}:
        raise ValueError("architecture must be 'lstm' or 'transformer'.")
    if n_features < 1 or n_classes < 2 or hidden_dim < 8 or num_layers < 1:
        raise ValueError("Invalid model dimensions.")

    class SequenceClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(n_features, hidden_dim)
            if architecture == "lstm":
                self.encoder = nn.LSTM(
                    hidden_dim,
                    hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                    bidirectional=False,
                )
            else:
                layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=4 if hidden_dim % 4 == 0 else 2 if hidden_dim % 2 == 0 else 1,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
                self.position = nn.Parameter(
                    torch.zeros(1, max_sequence_length, hidden_dim)
                )
                nn.init.normal_(self.position, mean=0.0, std=0.02)
            self.norm = nn.LayerNorm(hidden_dim)
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(hidden_dim, n_classes)

        def forward(self, values):
            encoded = self.projection(values)
            if architecture == "transformer":
                if encoded.shape[1] > self.position.shape[1]:
                    raise ValueError("Sequence exceeds max_sequence_length.")
                encoded = self.encoder(encoded + self.position[:, : encoded.shape[1]])
            else:
                encoded, _ = self.encoder(encoded)
            last = self.norm(encoded[:, -1, :])
            return self.head(self.dropout(last))

    return SequenceClassifier()


def train_sequence_model(
    model,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    epochs: int = 30,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    seed: int = 42,
    device: str | None = None,
) -> dict[str, Any]:
    """Train with class-weighted loss, gradient clipping and early stopping."""
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError("Install the optional ML dependencies: pip install -e '.[ml]'") from exc

    if train_x.ndim != 3 or validation_x.ndim != 3:
        raise ValueError("Sequence features must have shape [rows, steps, features].")
    if train_x.shape[1:] != validation_x.shape[1:]:
        raise ValueError("Training and validation sequence shapes must match.")
    if len(train_x) != len(train_y) or len(validation_x) != len(validation_y):
        raise ValueError("Sequence and label row counts must match.")
    if not len(train_x) or not len(validation_x):
        raise ValueError("Training and validation sequences must be non-empty.")
    if epochs < 1 or batch_size < 1 or patience < 1 or learning_rate <= 0:
        raise ValueError("Invalid training hyperparameters.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        selected_device = device or "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        selected_device = device or "mps"
    else:
        selected_device = device or "cpu"
    target_device = torch.device(selected_device)
    model = model.to(target_device)

    unique, counts = np.unique(train_y.astype(int), return_counts=True)
    class_map = {int(value): index for index, value in enumerate(unique)}
    if getattr(getattr(model, "head", None), "out_features", len(unique)) != len(unique):
        raise ValueError("Model output classes must match the training label classes.")
    if not set(np.unique(validation_y.astype(int))).issubset(class_map):
        raise ValueError("Validation labels contain a class absent from training.")
    mapped_train = np.asarray([class_map[int(value)] for value in train_y], dtype=np.int64)
    mapped_validation = np.asarray(
        [class_map[int(value)] for value in validation_y], dtype=np.int64
    )
    weights = len(train_y) / (len(unique) * counts.astype(np.float32))
    class_weights = torch.as_tensor(weights, device=target_device)
    train_dataset = TensorDataset(
        torch.as_tensor(train_x, dtype=torch.float32),
        torch.as_tensor(mapped_train, dtype=torch.long),
    )
    validation_values = torch.as_tensor(validation_x, dtype=torch.float32, device=target_device)
    validation_targets = torch.as_tensor(
        mapped_validation, dtype=torch.long, device=target_device
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=target_device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(target_device, non_blocking=True)
            batch_y = batch_y.to(target_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_y)
            total_rows += len(batch_y)

        model.eval()
        with torch.no_grad():
            validation_logits = model(validation_values)
            validation_loss = float(criterion(validation_logits, validation_targets))
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": total_loss / total_rows,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("Sequence training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    return {
        "model": model,
        "history": history,
        "best_validation_loss": best_loss,
        "epochs_completed": len(history),
        "device": str(target_device),
        "classes": [int(value) for value in unique],
    }
