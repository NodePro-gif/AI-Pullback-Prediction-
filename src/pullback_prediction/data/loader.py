"""Load Dukascopy-exported bid/ask OHLC candle data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ALIASES = {
    "timestamp": ("timestamp", "time", "datetime", "date"),
    "bid_open": ("bid_open", "open_bid", "bidopen", "open"),
    "bid_high": ("bid_high", "high_bid", "bidhigh", "high"),
    "bid_low": ("bid_low", "low_bid", "bidlow", "low"),
    "bid_close": ("bid_close", "close_bid", "bidclose", "close"),
    "ask_open": ("ask_open", "open_ask", "askopen"),
    "ask_high": ("ask_high", "high_ask", "askhigh"),
    "ask_low": ("ask_low", "low_ask", "asklow"),
    "ask_close": ("ask_close", "close_ask", "askclose"),
    "volume": ("volume", "tick_volume", "ticks", "vol"),
}

PRICE_COLUMNS = (
    "bid_open", "bid_high", "bid_low", "bid_close",
    "ask_open", "ask_high", "ask_low", "ask_close",
)


class MarketDataError(ValueError):
    """Raised when source candles violate the market-data contract."""


def _normalize_columns(columns: pd.Index) -> dict[str, str]:
    normalized = {str(column).strip().lower().replace(" ", "_"): str(column) for column in columns}
    mapping: dict[str, str] = {}
    for canonical, aliases in ALIASES.items():
        for alias in aliases:
            if alias in normalized:
                mapping[normalized[alias]] = canonical
                break
    return mapping


def load_dukascopy_csv(path: str | Path, timeframe: str) -> pd.DataFrame:
    """Read and validate bid/ask candles, returning a UTC DatetimeIndex.

    The loader accepts explicit bid/ask OHLC columns. Generic OHLC aliases are
    treated as bid prices only; ask prices must be present separately because
    executable spread cannot safely be inferred from bid-only data.
    """
    if timeframe not in {"M1", "M5"}:
        raise MarketDataError(f"Unsupported timeframe {timeframe!r}; expected M1 or M5.")

    source = pd.read_csv(path)
    mapping = _normalize_columns(source.columns)
    frame = source.rename(columns=mapping)

    required = {"timestamp", *PRICE_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise MarketDataError(
            "Missing required bid/ask fields: " + ", ".join(missing)
        )

    frame = frame[list(required) + (["volume"] if "volume" in frame.columns else [])].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if frame["timestamp"].isna().any():
        raise MarketDataError("Timestamp column contains invalid or missing values.")

    for column in PRICE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[list(PRICE_COLUMNS)].isna().any().any():
        raise MarketDataError("Bid/ask OHLC contains non-numeric or missing values.")

    if (frame[list(PRICE_COLUMNS)] <= 0).any().any():
        raise MarketDataError("Bid/ask prices must be positive.")

    invalid_bid = (
        (frame["bid_high"] < frame[["bid_open", "bid_close", "bid_low"]].max(axis=1))
        | (frame["bid_low"] > frame[["bid_open", "bid_close", "bid_high"]].min(axis=1))
    )
    invalid_ask = (
        (frame["ask_high"] < frame[["ask_open", "ask_close", "ask_low"]].max(axis=1))
        | (frame["ask_low"] > frame[["ask_open", "ask_close", "ask_high"]].min(axis=1))
    )
    if invalid_bid.any() or invalid_ask.any():
        raise MarketDataError("OHLC invariant violated: high/low does not contain open and close.")

    if (frame["ask_close"] < frame["bid_close"]).any():
        raise MarketDataError("Ask close is below bid close; check source columns.")

    if "volume" in frame.columns:
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")

    frame = frame.set_index("timestamp").sort_index()
    if frame.index.has_duplicates:
        raise MarketDataError("Duplicate candle timestamps detected.")
    expected = pd.Timedelta(minutes=1 if timeframe == "M1" else 5)
    gaps = frame.index.to_series().diff().dropna()
    if (gaps < expected).any():
        raise MarketDataError("Rows are more frequent than the selected timeframe.")

    return frame
