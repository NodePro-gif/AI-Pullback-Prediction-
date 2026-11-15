"""Read-only FastAPI inference service; no broker execution endpoints."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pullback_prediction.training.inference import predict_frame

MODEL_PATH = Path(os.getenv("PULLBACK_MODEL_PATH", "artifacts/xauusd_baseline.joblib"))
MIN_PROBABILITY = float(os.getenv("PULLBACK_MIN_PROBABILITY", "0.60"))

app = FastAPI(
    title="AI Pullback Prediction API",
    version="0.1.0",
    description="Read-only XAUUSD model inference. It never places broker orders.",
)


class Candle(BaseModel):
    timestamp: datetime
    bid_open: float = Field(gt=0, allow_inf_nan=False)
    bid_high: float = Field(gt=0, allow_inf_nan=False)
    bid_low: float = Field(gt=0, allow_inf_nan=False)
    bid_close: float = Field(gt=0, allow_inf_nan=False)
    ask_open: float = Field(gt=0, allow_inf_nan=False)
    ask_high: float = Field(gt=0, allow_inf_nan=False)
    ask_low: float = Field(gt=0, allow_inf_nan=False)
    ask_close: float = Field(gt=0, allow_inf_nan=False)
    volume: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class PredictionRequest(BaseModel):
    candles: list[Candle] = Field(min_length=220, max_length=5000)


class Prediction(BaseModel):
    timestamp: datetime
    signal: str
    confidence: float
    prob_sell: float
    prob_wait: float
    prob_buy: float
    atr_14: float | None


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_configured": MODEL_PATH.is_file(),
        "live_execution_enabled": False,
    }


@app.post("/predict", response_model=list[Prediction])
def predict(request: PredictionRequest) -> list[Prediction]:
    if not MODEL_PATH.is_file():
        raise HTTPException(status_code=503, detail="Model artifact is not configured.")
    frame = pd.DataFrame([item.dict() for item in request.candles])
    frame = frame.set_index("timestamp")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise HTTPException(status_code=422, detail="Timestamps must include a timezone.")
    if frame.index.has_duplicates:
        raise HTTPException(status_code=422, detail="Candle timestamps must be unique.")
    if not frame.index.is_monotonic_increasing:
        raise HTTPException(status_code=422, detail="Candle timestamps must be chronological.")
    frame.index = frame.index.tz_convert("UTC")
    price_columns = [column for column in frame if column != "volume"]
    if not np.isfinite(frame[price_columns].to_numpy(dtype=float)).all():
        raise HTTPException(status_code=422, detail="Candle prices must be finite.")
    for side in ("bid", "ask"):
        expected_high = frame[[f"{side}_open", f"{side}_close", f"{side}_low"]].max(axis=1)
        expected_low = frame[[f"{side}_open", f"{side}_close", f"{side}_high"]].min(axis=1)
        if (
            (frame[f"{side}_high"] < expected_high).any()
            or (frame[f"{side}_low"] > expected_low).any()
        ):
            raise HTTPException(status_code=422, detail=f"Invalid {side} OHLC candle.")
    for field in ("open", "high", "low", "close"):
        if (frame[f"ask_{field}"] < frame[f"bid_{field}"]).any():
            raise HTTPException(status_code=422, detail="Ask prices cannot be below bid prices.")
    try:
        result = predict_frame(frame, MODEL_PATH, min_probability=MIN_PROBABILITY)
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    names = {-1: "SELL", 0: "WAIT", 1: "BUY"}
    rows: list[Prediction] = []
    for timestamp, row in result.iterrows():
        atr_value = row["atr_14"]
        rows.append(
            Prediction(
                timestamp=timestamp.to_pydatetime(),
                signal=names[int(row["signal"])],
                confidence=float(row["confidence"]),
                prob_sell=float(row["prob_sell"]),
                prob_wait=float(row["prob_wait"]),
                prob_buy=float(row["prob_buy"]),
                atr_14=None if pd.isna(atr_value) else float(atr_value),
            )
        )
    return rows
