import pandas as pd
import pytest

from pullback_prediction.backtesting.engine import backtest_signals


def candles(rows, columns):
    index = pd.date_range("2024-01-01", periods=len(rows), freq="min", tz="UTC")
    return pd.DataFrame(rows, index=index, columns=columns)


def test_long_trade_enters_next_open_and_accounts_for_spread_slippage_and_commission():
    frame = candles(
        [
            [100.0, 100.5, 99.5, 100.0, 100.2, 100.7, 99.7, 100.2],
            [100.8, 102.0, 100.5, 101.8, 101.0, 102.2, 100.7, 102.0],
            [102.0, 104.0, 102.5, 103.5, 102.2, 104.2, 102.7, 103.7],
            [103.5, 104.0, 103.0, 103.5, 103.7, 104.2, 103.2, 103.7],
        ],
        columns=[
            "bid_open", "bid_high", "bid_low", "bid_close",
            "ask_open", "ask_high", "ask_low", "ask_close",
        ],
    )
    signals = pd.Series([1, 0, 0, 0], index=frame.index)
    atr = pd.Series(1.0, index=frame.index)

    result = backtest_signals(
        frame,
        signals,
        atr,
        stop_atr_multiple=1.0,
        take_profit_atr_multiple=2.0,
        max_holding_bars=3,
        slippage_price=0.1,
        commission_price_units=0.2,
    )

    trade = result["trades"].iloc[0]
    assert trade["entry_time"] == frame.index[1]
    assert trade["exit_time"] == frame.index[2]
    assert trade["entry_price"] == pytest.approx(101.1)
    assert trade["exit_price"] == pytest.approx(103.0)
    assert trade["exit_reason"] == "take_profit"
    assert trade["net_pnl"] == pytest.approx(1.7)


def test_same_bar_stop_and_target_uses_conservative_stop_fill():
    frame = candles(
        [
            [100.0, 100.5, 99.5, 100.0, 100.2, 100.7, 99.7, 100.2],
            [100.8, 103.0, 99.0, 101.0, 101.0, 103.2, 99.2, 101.2],
        ],
        columns=[
            "bid_open", "bid_high", "bid_low", "bid_close",
            "ask_open", "ask_high", "ask_low", "ask_close",
        ],
    )
    result = backtest_signals(
        frame,
        pd.Series([1, 0], index=frame.index),
        pd.Series(1.0, index=frame.index),
        stop_atr_multiple=1.0,
        take_profit_atr_multiple=2.0,
        max_holding_bars=1,
    )

    trade = result["trades"].iloc[0]
    assert trade["exit_reason"] == "stop_loss"
    assert trade["net_pnl"] == pytest.approx(-1.0)


def test_backtester_rejects_misaligned_inputs():
    frame = candles(
        [[100, 101, 99, 100, 100.2, 101.2, 99.2, 100.2]],
        columns=[
            "bid_open", "bid_high", "bid_low", "bid_close",
            "ask_open", "ask_high", "ask_low", "ask_close",
        ],
    )
    with pytest.raises(ValueError, match="same index"):
        backtest_signals(
            frame,
            pd.Series([1], index=frame.index),
            pd.Series([1.0], index=pd.date_range("2024-01-02", periods=1, tz="UTC")),
        )
