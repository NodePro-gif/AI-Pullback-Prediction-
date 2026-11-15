import json

import pandas as pd
from typer.testing import CliRunner

from pullback_prediction.cli import app


def test_backtest_cli_writes_trade_and_summary_files(tmp_path):
    index = pd.date_range("2024-01-01", periods=2, freq="min", tz="UTC")
    frame = pd.DataFrame(
        {
            "bid_open": [100.0, 100.8],
            "bid_high": [100.5, 104.0],
            "bid_low": [99.5, 100.5],
            "bid_close": [100.0, 103.5],
            "ask_open": [100.2, 101.0],
            "ask_high": [100.7, 104.2],
            "ask_low": [99.7, 100.7],
            "ask_close": [100.2, 103.7],
            "atr_14": [1.0, 1.0],
            "signal": [1, 0],
        },
        index=index,
    )
    source = tmp_path / "signals.parquet"
    output_dir = tmp_path / "backtest"
    frame.to_parquet(source)

    result = CliRunner().invoke(
        app,
        [
            "backtest-signals",
            "--input",
            str(source),
            "--output-dir",
            str(output_dir),
            "--max-holding-bars",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output_dir / "trades.csv").exists()
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["total_trades"] == 1
    assert summary["net_pnl"] > 0



def test_prepare_data_cli_converts_bid_ask_csv_to_parquet(tmp_path):
    bid_path = tmp_path / "bid.csv"
    ask_path = tmp_path / "ask.csv"
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
    output = tmp_path / "processed" / "xauusd_m1.parquet"

    result = CliRunner().invoke(
        app,
        [
            "prepare-data",
            "--bid-input",
            str(bid_path),
            "--ask-input",
            str(ask_path),
            "--timeframe",
            "M1",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    prepared = pd.read_parquet(output)
    assert len(prepared) == 2
    assert prepared.index[0] == pd.Timestamp("2024-01-01T00:00:00Z")
    assert prepared["ask_close"].gt(prepared["bid_close"]).all()
