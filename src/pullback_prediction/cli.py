"""Command-line interface for data preparation, training, prediction, and evaluation."""

import json
from pathlib import Path

import pandas as pd
import typer

from pullback_prediction.backtesting.engine import backtest_signals
from pullback_prediction.data.downloader import download_dukascopy_bid_ask
from pullback_prediction.data.loader import (
    load_dukascopy_bid_ask_pair,
    load_dukascopy_csv,
)
from pullback_prediction.data.splits import chronological_split
from pullback_prediction.training.baseline import train_baseline_from_frame
from pullback_prediction.training.inference import predict_frame

app = typer.Typer(no_args_is_help=True, help="XAUUSD pullback prediction research tools.")


@app.command("download-data")
def download_data_command(
    from_date: str = typer.Option(..., "--from-date"),
    to_date: str = typer.Option(..., "--to-date"),
    timeframe: str = typer.Option(..., "--timeframe"),
    output_dir: Path = typer.Option(Path("data/raw"), "--output-dir"),
    batch_size: int = typer.Option(10, "--batch-size", min=1),
    batch_pause_ms: int = typer.Option(1000, "--batch-pause-ms", min=0),
    retries: int = typer.Option(3, "--retries", min=0),
) -> None:
    """Download raw XAUUSD bid and ask CSV exports from Dukascopy."""
    bid_path, ask_path = download_dukascopy_bid_ask(
        from_date,
        to_date,
        timeframe=timeframe,
        output_dir=output_dir,
        batch_size=batch_size,
        batch_pause_ms=batch_pause_ms,
        retries=retries,
    )
    typer.echo(f"bid_file={bid_path}")
    typer.echo(f"ask_file={ask_path}")

@app.command("prepare-data")
def prepare_data_command(
    bid_input: Path = typer.Option(..., "--bid-input", exists=True, readable=True),
    ask_input: Path = typer.Option(..., "--ask-input", exists=True, readable=True),
    timeframe: str = typer.Option(..., "--timeframe"),
    output: Path = typer.Option(..., "--output"),
) -> None:
    """Validate and combine separate bid/ask CSV files into Parquet."""
    if output.exists():
        raise typer.BadParameter(f"Refusing to overwrite processed data: {output}")
    frame = load_dukascopy_bid_ask_pair(bid_input, ask_input, timeframe=timeframe)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output)
    typer.echo(f"rows={len(frame)}")
    typer.echo(f"start={frame.index.min().isoformat()}")
    typer.echo(f"end={frame.index.max().isoformat()}")
    typer.echo(f"output={output}")

@app.command("validate-data")
def validate_data(
    input: Path = typer.Option(..., "--input", exists=True, readable=True),
    timeframe: str = typer.Option(..., "--timeframe"),
    ask_input: Path | None = typer.Option(None, "--ask-input", exists=True, readable=True),
) -> None:
    """Validate a combined bid/ask CSV or separate bid and ask CSV exports."""
    if ask_input is None:
        frame = load_dukascopy_csv(input, timeframe=timeframe)
    else:
        frame = load_dukascopy_bid_ask_pair(input, ask_input, timeframe=timeframe)
    typer.echo(f"rows={len(frame)}")
    typer.echo(f"start={frame.index.min().isoformat()}")
    typer.echo(f"end={frame.index.max().isoformat()}")
    typer.echo(f"columns={','.join(frame.columns)}")


@app.command("split-data")
def split_data(
    input: Path = typer.Option(..., "--input", exists=True, readable=True),
    output_dir: Path = typer.Option(..., "--output-dir"),
) -> None:
    """Write fixed chronological train, validation and final-test partitions."""
    frame = pd.read_parquet(input)
    partitions = chronological_split(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, partition in partitions.items():
        target = output_dir / f"{name}.parquet"
        partition.to_parquet(target)
        typer.echo(f"{name}: rows={len(partition)} path={target}")


@app.command("train-baseline")
def train_baseline(
    input: Path = typer.Option(..., "--input", exists=True, readable=True),
    output: Path = typer.Option(Path("artifacts/xauusd_baseline.joblib"), "--output"),
    horizon_bars: int = typer.Option(12, "--horizon-bars", min=1),
    barrier_atr_multiple: float = typer.Option(1.5, "--barrier-atr-multiple", min=0.01),
    max_iter: int = typer.Option(200, "--max-iter", min=1),
) -> None:
    """Train a gradient-boosting baseline; final-test data is excluded."""
    frame = pd.read_parquet(input)
    result = train_baseline_from_frame(
        frame,
        output,
        horizon_bars=horizon_bars,
        barrier_atr_multiple=barrier_atr_multiple,
        max_iter=max_iter,
    )
    for key, value in result.items():
        typer.echo(f"{key}={value}")


@app.command("predict")
def predict_command(
    input: Path = typer.Option(..., "--input", exists=True, readable=True),
    model: Path = typer.Option(..., "--model", exists=True, readable=True),
    output: Path = typer.Option(Path("data/processed/predictions.parquet"), "--output"),
    min_probability: float = typer.Option(0.60, "--min-probability", min=0.0, max=1.0),
) -> None:
    """Generate BUY/SELL/WAIT signals and class scores from a saved model."""
    frame = pd.read_parquet(input)
    predictions = predict_frame(frame, model, min_probability=min_probability)
    overlapping = [column for column in predictions.columns if column in frame.columns]
    output_frame = frame.drop(columns=overlapping).join(predictions)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_parquet(output)
    typer.echo(f"rows={len(output_frame)}")
    typer.echo(f"buy_signals={int((predictions['signal'] == 1).sum())}")
    typer.echo(f"sell_signals={int((predictions['signal'] == -1).sum())}")
    typer.echo(f"wait_signals={int((predictions['signal'] == 0).sum())}")
    typer.echo(f"output={output}")


@app.command("backtest-signals")
def backtest_signals_command(
    input: Path = typer.Option(..., "--input", exists=True, readable=True),
    output_dir: Path = typer.Option(Path("artifacts/backtest"), "--output-dir"),
    signal_column: str = typer.Option("signal", "--signal-column"),
    atr_column: str = typer.Option("atr_14", "--atr-column"),
    stop_atr_multiple: float = typer.Option(1.5, "--stop-atr-multiple", min=0.01),
    take_profit_atr_multiple: float = typer.Option(
        2.0, "--take-profit-atr-multiple", min=0.01
    ),
    max_holding_bars: int = typer.Option(12, "--max-holding-bars", min=1),
    slippage_price: float = typer.Option(0.0, "--slippage-price", min=0.0),
    commission_price_units: float = typer.Option(
        0.0, "--commission-price-units", min=0.0
    ),
) -> None:
    """Backtest -1/0/+1 signals with bid/ask fills and explicit costs."""
    frame = pd.read_parquet(input)
    missing = {signal_column, atr_column}.difference(frame.columns)
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise typer.BadParameter(f"Missing signal/ATR columns: {missing_names}")
    signals = frame[signal_column].astype("int8")
    atr = frame[atr_column].astype(float)
    candles = frame.drop(columns=[signal_column, atr_column])
    result = backtest_signals(
        candles,
        signals,
        atr,
        stop_atr_multiple=stop_atr_multiple,
        take_profit_atr_multiple=take_profit_atr_multiple,
        max_holding_bars=max_holding_bars,
        slippage_price=slippage_price,
        commission_price_units=commission_price_units,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    trades_path = output_dir / "trades.csv"
    summary_path = output_dir / "summary.json"
    result["trades"].to_csv(trades_path, index=False)
    summary_path.write_text(
        json.dumps(result["summary"], indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    typer.echo(f"trades_path={trades_path}")
    typer.echo(f"summary_path={summary_path}")
    for key, value in result["summary"].items():
        typer.echo(f"{key}={value}")


if __name__ == "__main__":
    app()
