"""Command-line interface for data validation, preparation, and baseline training."""

from pathlib import Path

import pandas as pd
import typer

from pullback_prediction.data.loader import (
    load_dukascopy_bid_ask_pair,
    load_dukascopy_csv,
)
from pullback_prediction.data.splits import chronological_split
from pullback_prediction.training.baseline import train_baseline_from_frame

app = typer.Typer(no_args_is_help=True, help="XAUUSD pullback prediction research tools.")


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


if __name__ == "__main__":
    app()
