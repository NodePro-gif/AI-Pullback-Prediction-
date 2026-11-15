import numpy as np
import pandas as pd
import pytest

from pullback_prediction.api import app as api


def make_request(timeframe="M1"):
    frequency = "min" if timeframe == "M1" else "5min"
    timestamps = pd.date_range("2024-01-01", periods=220, freq=frequency, tz="UTC")
    candles = []
    for timestamp in timestamps:
        row = {
            "timestamp": timestamp.to_pydatetime(),
            "bid_open": 2000.0,
            "bid_high": 2001.0,
            "bid_low": 1999.0,
            "bid_close": 2000.5,
            "ask_open": 2000.2,
            "ask_high": 2001.2,
            "ask_low": 1999.2,
            "ask_close": 2000.7,
            "volume": None,
        }
        candles.append(api.Candle(**row))
    return api.PredictionRequest(timeframe=timeframe, candles=candles)


def test_api_health_never_enables_live_execution(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "missing.joblib")
    result = api.health()
    assert result["status"] == "ok"
    assert result["model_configured"] is False
    assert result["live_execution_enabled"] is False


def test_api_predict_rejects_unsorted_timestamps(monkeypatch, tmp_path):
    model = tmp_path / "model.joblib"
    model.touch()
    monkeypatch.setattr(api, "MODEL_PATH", model)
    request = make_request()
    request.candles[5], request.candles[6] = request.candles[6], request.candles[5]

    with pytest.raises(api.HTTPException) as error:
        api.predict(request)

    assert error.value.status_code == 422
    assert "chronological" in error.value.detail


def test_api_predict_returns_buy_sell_wait_labels(monkeypatch, tmp_path):
    model = tmp_path / "model.joblib"
    model.touch()
    monkeypatch.setattr(api, "MODEL_PATH", model)

    def fake_predict(frame, artifact_path, min_probability, timeframe):
        assert timeframe == "M1"
        return pd.DataFrame(
            {
                "signal": np.zeros(len(frame), dtype="int8"),
                "confidence": np.full(len(frame), 0.5),
                "probabilities_calibrated": np.ones(len(frame), dtype=bool),
                "prob_sell": np.full(len(frame), 0.2),
                "prob_wait": np.full(len(frame), 0.5),
                "prob_buy": np.full(len(frame), 0.3),
                "atr_14": np.ones(len(frame)),
            },
            index=frame.index,
        )

    monkeypatch.setattr(api, "predict_frame", fake_predict)
    result = api.predict(make_request())
    assert len(result) == 220
    assert result[-1].signal == "WAIT"
    assert result[-1].prob_wait == 0.5
    assert result[-1].probabilities_calibrated is True


def test_api_predict_rejects_naive_timestamps(monkeypatch, tmp_path):
    model = tmp_path / "model.joblib"
    model.touch()
    monkeypatch.setattr(api, "MODEL_PATH", model)
    request = make_request()
    request.candles[0].timestamp = pd.Timestamp("2024-01-01").to_pydatetime()

    with pytest.raises(api.HTTPException) as error:
        api.predict(request)

    assert error.value.status_code == 422
    assert "timezone" in error.value.detail.lower()


def test_api_predict_rejects_crossed_bid_ask_quotes(monkeypatch, tmp_path):
    model = tmp_path / "model.joblib"
    model.touch()
    monkeypatch.setattr(api, "MODEL_PATH", model)
    request = make_request()
    request.candles[5].ask_close = 2000.0

    with pytest.raises(api.HTTPException) as error:
        api.predict(request)

    assert error.value.status_code == 422
    assert "ask" in error.value.detail.lower()


def test_api_predict_rejects_invalid_ohlc_ranges(monkeypatch, tmp_path):
    model = tmp_path / "model.joblib"
    model.touch()
    monkeypatch.setattr(api, "MODEL_PATH", model)
    request = make_request()
    request.candles[5].bid_high = 1999.0

    with pytest.raises(api.HTTPException) as error:
        api.predict(request)

    assert error.value.status_code == 422
    assert "ohlc" in error.value.detail.lower()

def test_api_predict_rejects_ask_below_bid(monkeypatch, tmp_path):
    model = tmp_path / "model.joblib"
    model.touch()
    monkeypatch.setattr(api, "MODEL_PATH", model)
    request = make_request()
    request.candles[0].ask_open = 1999.2
    request.candles[0].ask_high = 2000.2
    request.candles[0].ask_low = 1998.2
    request.candles[0].ask_close = 1999.7

    with pytest.raises(api.HTTPException) as error:
        api.predict(request)

    assert error.value.status_code == 422
    assert "Ask prices" in error.value.detail



def test_api_rejects_candle_spacing_that_disagrees_with_timeframe(monkeypatch, tmp_path):
    model = tmp_path / "model.joblib"
    model.touch()
    monkeypatch.setattr(api, "MODEL_PATH", model)
    request = make_request("M1")
    request.timeframe = "M5"

    with pytest.raises(api.HTTPException) as error:
        api.predict(request)

    assert error.value.status_code == 422
    assert "spacing" in error.value.detail.lower()
