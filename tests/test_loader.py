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


def test_loader_aligns_separate_bid_and_ask_exports(tmp_path):
    bid_path = tmp_path / "bid.csv"
    ask_path = tmp_path / "ask.csv"
    base = {
        "Gmt time": ["2020-01-01T00:00:00Z", "2020-01-01T00:01:00Z"],
        "Open": [100.0, 100.1],
        "High": [100.5, 100.6],
        "Low": [99.8, 100.0],
        "Close": [100.2, 100.3],
    }
    pd.DataFrame(base).to_csv(bid_path, index=False)
    ask = {**base, "Open": [100.1, 100.2], "High": [100.6, 100.7],
           "Low": [99.9, 100.1], "Close": [100.3, 100.4]}
    pd.DataFrame(ask).to_csv(ask_path, index=False)

    from pullback_prediction.data.loader import load_dukascopy_bid_ask_pair

    result = load_dukascopy_bid_ask_pair(bid_path, ask_path, "M1")
    assert len(result) == 2
    assert result["ask_close"].gt(result["bid_close"]).all()


def test_loader_rejects_misaligned_bid_ask_timestamps(tmp_path):
    bid_path = tmp_path / "bid.csv"
    ask_path = tmp_path / "ask.csv"
    bid = {"timestamp": ["2020-01-01T00:00:00Z"], "open": [100], "high": [101],
           "low": [99], "close": [100]}
    ask = {"timestamp": ["2020-01-01T00:01:00Z"], "open": [101], "high": [102],
           "low": [100], "close": [101]}
    pd.DataFrame(bid).to_csv(bid_path, index=False)
    pd.DataFrame(ask).to_csv(ask_path, index=False)

    from pullback_prediction.data.loader import load_dukascopy_bid_ask_pair

    with pytest.raises(MarketDataError, match="timestamps do not match exactly"):
        load_dukascopy_bid_ask_pair(bid_path, ask_path, "M1")



def test_loader_interprets_dukascopy_unix_milliseconds_as_utc(tmp_path):
    path = tmp_path / "milliseconds.csv"
    pd.DataFrame(
        {
            "timestamp": [1704067200000, 1704067260000],
            "bid_open": [100.0, 100.1],
            "bid_high": [100.5, 100.6],
            "bid_low": [99.8, 100.0],
            "bid_close": [100.2, 100.3],
            "ask_open": [100.1, 100.2],
            "ask_high": [100.6, 100.7],
            "ask_low": [99.9, 100.1],
            "ask_close": [100.3, 100.4],
        }
    ).to_csv(path, index=False)

    result = load_dukascopy_csv(path, "M1")

    assert result.index[0] == pd.Timestamp("2024-01-01T00:00:00Z")
    assert result.index[1] == pd.Timestamp("2024-01-01T00:01:00Z")


def test_paired_loader_interprets_unix_milliseconds_as_utc(tmp_path):
    from pullback_prediction.data.loader import load_dukascopy_bid_ask_pair

    bid_path = tmp_path / "bid-ms.csv"
    ask_path = tmp_path / "ask-ms.csv"
    base = {
        "timestamp": [1704067200000, 1704067260000],
        "open": [100.0, 100.1],
        "high": [100.5, 100.6],
        "low": [99.8, 100.0],
        "close": [100.2, 100.3],
    }
    pd.DataFrame(base).to_csv(bid_path, index=False)
    ask = {
        **base,
        "open": [100.1, 100.2],
        "high": [100.6, 100.7],
        "low": [99.9, 100.1],
        "close": [100.3, 100.4],
    }
    pd.DataFrame(ask).to_csv(ask_path, index=False)

    result = load_dukascopy_bid_ask_pair(bid_path, ask_path, "M1")

    assert result.index[0] == pd.Timestamp("2024-01-01T00:00:00Z")
    assert len(result) == 2
