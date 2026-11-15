"""Single-position bid/ask backtesting with conservative intrabar fills."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_REQUIRED = {
    "bid_open", "bid_high", "bid_low", "bid_close",
    "ask_open", "ask_high", "ask_low", "ask_close",
}


def _summary(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "net_pnl": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "max_drawdown": 0.0,
        }
    pnl = trades["net_pnl"]
    gains = float(pnl[pnl > 0].sum())
    losses = float(-pnl[pnl < 0].sum())
    equity = pnl.cumsum()
    drawdown = equity - equity.cummax()
    profit_factor = gains / losses if losses > 0 else None
    return {
        "total_trades": int(len(trades)),
        "win_rate": float((pnl > 0).mean()),
        "net_pnl": float(pnl.sum()),
        "profit_factor": profit_factor,
        "expectancy": float(pnl.mean()),
        "max_drawdown": float(drawdown.min()),
    }


def backtest_signals(
    candles: pd.DataFrame,
    signals: pd.Series,
    atr: pd.Series,
    *,
    stop_atr_multiple: float = 1.5,
    take_profit_atr_multiple: float = 2.0,
    max_holding_bars: int = 12,
    slippage_price: float = 0.0,
    commission_price_units: float = 0.0,
) -> dict[str, Any]:
    """Simulate close-generated signals at the next candle open.

    Signal values are -1 (short), 0 (flat), and +1 (long). Longs enter at ask
    and exit at bid; shorts enter at bid and exit at ask. Slippage is charged
    adversely on both entry and exit. If stop and target are both touched in
    one candle, the stop is assumed to fill first. Commission is a total
    round-trip deduction expressed in price units per trade.

    This is a research simulator, not a broker execution model. PnL is in
    instrument price units per unit, not account currency or lot sizing.
    """
    missing = _REQUIRED.difference(candles.columns)
    if missing:
        raise ValueError(f"Missing bid/ask OHLC columns: {', '.join(sorted(missing))}")
    if not candles.index.equals(signals.index) or not candles.index.equals(atr.index):
        raise ValueError("Candles, signals, and ATR must have the same index.")
    if not isinstance(candles.index, pd.DatetimeIndex):
        raise TypeError("Candles must use a DatetimeIndex.")
    if not candles.index.is_monotonic_increasing or candles.index.has_duplicates:
        raise ValueError("Candle timestamps must be unique and sorted ascending.")
    if not signals.isin([-1, 0, 1]).all():
        raise ValueError("Signals must contain only -1, 0, or 1.")
    if stop_atr_multiple <= 0 or take_profit_atr_multiple <= 0:
        raise ValueError("Stop and take-profit ATR multiples must be positive.")
    if max_holding_bars < 1:
        raise ValueError("max_holding_bars must be at least 1.")
    if slippage_price < 0 or commission_price_units < 0:
        raise ValueError("Slippage and commission cannot be negative.")

    trades: list[dict[str, Any]] = []
    signal_values = signals.to_numpy(dtype=np.int8)
    atr_values = atr.to_numpy(dtype=float)
    bid_open = candles["bid_open"].to_numpy(dtype=float)
    bid_high = candles["bid_high"].to_numpy(dtype=float)
    bid_low = candles["bid_low"].to_numpy(dtype=float)
    bid_close = candles["bid_close"].to_numpy(dtype=float)
    ask_open = candles["ask_open"].to_numpy(dtype=float)
    ask_high = candles["ask_high"].to_numpy(dtype=float)
    ask_low = candles["ask_low"].to_numpy(dtype=float)
    ask_close = candles["ask_close"].to_numpy(dtype=float)

    signal_i = 0
    while signal_i < len(candles) - 1:
        side = int(signal_values[signal_i])
        risk_distance = atr_values[signal_i] * stop_atr_multiple
        if side == 0 or not np.isfinite(risk_distance) or risk_distance <= 0:
            signal_i += 1
            continue

        entry_i = signal_i + 1
        if side == 1:
            entry_price = ask_open[entry_i] + slippage_price
        else:
            entry_price = bid_open[entry_i] - slippage_price
        if not np.isfinite(entry_price):
            signal_i += 1
            continue

        stop_price = entry_price - side * risk_distance
        target_price = entry_price + side * (
            atr_values[signal_i] * take_profit_atr_multiple
        )
        scan_end = min(entry_i + max_holding_bars - 1, len(candles) - 1)
        exit_price = float("nan")
        exit_i = scan_end
        exit_reason = "time_exit"

        for bar_i in range(entry_i, scan_end + 1):
            if side == 1:
                open_price = bid_open[bar_i]
                high_price = bid_high[bar_i]
                low_price = bid_low[bar_i]
                if open_price <= stop_price:
                    exit_price = open_price - slippage_price
                    exit_reason = "stop_loss"
                elif open_price >= target_price:
                    exit_price = open_price - slippage_price
                    exit_reason = "take_profit"
                elif low_price <= stop_price and high_price >= target_price:
                    exit_price = stop_price - slippage_price
                    exit_reason = "stop_loss"
                elif low_price <= stop_price:
                    exit_price = stop_price - slippage_price
                    exit_reason = "stop_loss"
                elif high_price >= target_price:
                    exit_price = target_price - slippage_price
                    exit_reason = "take_profit"
            else:
                open_price = ask_open[bar_i]
                high_price = ask_high[bar_i]
                low_price = ask_low[bar_i]
                if open_price >= stop_price:
                    exit_price = open_price + slippage_price
                    exit_reason = "stop_loss"
                elif open_price <= target_price:
                    exit_price = open_price + slippage_price
                    exit_reason = "take_profit"
                elif high_price >= stop_price and low_price <= target_price:
                    exit_price = stop_price + slippage_price
                    exit_reason = "stop_loss"
                elif high_price >= stop_price:
                    exit_price = stop_price + slippage_price
                    exit_reason = "stop_loss"
                elif low_price <= target_price:
                    exit_price = target_price + slippage_price
                    exit_reason = "take_profit"
            if np.isfinite(exit_price):
                exit_i = bar_i
                break

        if not np.isfinite(exit_price):
            exit_price = (
                bid_close[exit_i] - slippage_price
                if side == 1
                else ask_close[exit_i] + slippage_price
            )
        gross_pnl = side * (exit_price - entry_price)
        net_pnl = gross_pnl - commission_price_units
        trades.append(
            {
                "signal_time": candles.index[signal_i],
                "entry_time": candles.index[entry_i],
                "exit_time": candles.index[exit_i],
                "side": "BUY" if side == 1 else "SELL",
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "gross_pnl": float(gross_pnl),
                "commission": float(commission_price_units),
                "net_pnl": float(net_pnl),
                "risk_per_unit": float(risk_distance),
                "r_multiple": float(net_pnl / risk_distance),
                "bars_held": int(exit_i - entry_i + 1),
                "exit_reason": exit_reason,
            }
        )
        signal_i = exit_i + 1

    trade_frame = pd.DataFrame(trades)
    return {"trades": trade_frame, "summary": _summary(trade_frame)}
