import pandas as pd
import pytest

from pullback_prediction.data.loader import MarketDataError, load_dukascopy_csv


def _valid_csv(path):
    pd.DataFrame(
        {
            "timestamp": ["2020-01-01T00:00:00Z", "2020-01-01T00:01:00Z"],
            "bid_open": [100.0, 100.1],
            "bid_high": [100.5, 100.6],
            "bid_low": [99.8, 100.0],
            "bid_close": [100.2, 100.3],
            "ask_open": [100.1, 100.2],
            "ask_high": [100.6, 100.7],
            "ask_low": [99.9, 100.1],
            "ask_close": [100.3, 100.4],
            "volume": [10, 12],
        }
    ).to_csv(path, index=False)


def test_loader_returns_sorted_utc_bid_ask_frame(tmp_path):
    path = tmp_path / "candles.csv"
    _valid_csv(path)
    result = load_dukascopy_csv(path, "M1")
    assert str(result.index.tz) == "UTC"
    assert "bid_close" in result
    assert "ask_close" in result
    assert result.iloc[0]["ask_close"] >= result.iloc[0]["bid_close"]


def test_loader_rejects_bid_only_input(tmp_path):
    path = tmp_path / "bid_only.csv"
    pd.DataFrame(
        {
            "timestamp": ["2020-01-01T00:00:00Z"],
            "open": [100],
            "high": [101],
            "low": [99],
            "close": [100],
        }
    ).to_csv(path, index=False)
    with pytest.raises(MarketDataError, match="Missing required bid/ask fields"):
        load_dukascopy_csv(path, "M1")


def test_loader_rejects_invalid_timeframe(tmp_path):
    path = tmp_path / "unused.csv"
    with pytest.raises(MarketDataError, match="Unsupported timeframe"):
        load_dukascopy_csv(path, "M15")
