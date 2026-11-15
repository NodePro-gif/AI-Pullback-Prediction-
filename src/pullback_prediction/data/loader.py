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


def load_dukascopy_bid_ask_pair(
    bid_path: str | Path,
    ask_path: str | Path,
    timeframe: str,
) -> pd.DataFrame:
    """Load separate Dukascopy bid and ask OHLC exports and align by timestamp.

    Each file may use generic OHLC names (Open/High/Low/Close) or explicit
    side-specific names. A timestamp mismatch is rejected instead of silently
    forward-filling one side.
    """
    if timeframe not in {"M1", "M5"}:
        raise MarketDataError(f"Unsupported timeframe {timeframe!r}; expected M1 or M5.")

    def read_side(path: str | Path, side: str) -> pd.DataFrame:
        raw = pd.read_csv(path)
        normalized = {
            str(column).strip().lower().replace(" ", "_"): str(column)
            for column in raw.columns
        }
        time_aliases = ("timestamp", "time", "datetime", "date", "gmt_time")
        time_column = next((normalized[name] for name in time_aliases if name in normalized), None)
        if time_column is None:
            raise MarketDataError(f"{side} file has no recognized timestamp column.")

        side_mapping = {}
        for field in ("open", "high", "low", "close"):
            aliases = (
                (f"{side}_{field}", f"{field}_{side}", f"{side}{field}", field)
            )
            source_column = next(
                (normalized[name] for name in aliases if name in normalized), None
            )
            if source_column is None:
                raise MarketDataError(f"{side} file is missing {field} price.")
            side_mapping[source_column] = f"{side}_{field}"

        result = raw[[time_column, *side_mapping]].rename(
            columns={time_column: "timestamp", **side_mapping}
        )
        result["timestamp"] = pd.to_datetime(
            result["timestamp"], utc=True, errors="coerce"
        )
        if result["timestamp"].isna().any():
            raise MarketDataError(f"{side} file contains invalid timestamps.")
        price_columns = [f"{side}_{field}" for field in ("open", "high", "low", "close")]
        for column in price_columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
        if result[price_columns].isna().any().any() or (result[price_columns] <= 0).any().any():
            raise MarketDataError(f"{side} file contains missing, non-numeric, or non-positive prices.")
        invalid = (
            (result[f"{side}_high"] < result[[f"{side}_open", f"{side}_close", f"{side}_low"]].max(axis=1))
            | (result[f"{side}_low"] > result[[f"{side}_open", f"{side}_close", f"{side}_high"]].min(axis=1))
        )
        if invalid.any():
            raise MarketDataError(f"{side} OHLC invariant violated.")
        result = result.set_index("timestamp").sort_index()
        if result.index.has_duplicates:
            raise MarketDataError(f"{side} file contains duplicate timestamps.")
        return result

    bid = read_side(bid_path, "bid")
    ask = read_side(ask_path, "ask")
    if not bid.index.equals(ask.index):
        raise MarketDataError("Bid and ask timestamps do not match exactly.")

    result = bid.join(ask, how="inner")
    if (result["ask_close"] < result["bid_close"]).any():
        raise MarketDataError("Ask close is below bid close; check source files.")

    expected = pd.Timedelta(minutes=1 if timeframe == "M1" else 5)
    gaps = result.index.to_series().diff().dropna()
    if (gaps < expected).any():
        raise MarketDataError("Rows are more frequent than the selected timeframe.")
    return result
