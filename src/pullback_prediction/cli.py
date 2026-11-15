"""Command-line interface for data preparation, training, prediction, and evaluation."""

import json
from pathlib import Path

import joblib
import pandas as pd
import typer

from pullback_prediction.backtesting.engine import backtest_signals
from pullback_prediction.data.downloader import download_dukascopy_bid_ask
from pullback_prediction.data.loader import (
    load_dukascopy_bid_ask_pair,
    load_dukascopy_csv,
)
from pullback_prediction.data.splits import chronological_split
from pullback_prediction.features.technical import build_features
from pullback_prediction.labels.barriers import make_barrier_labels
from pullback_prediction.training.baseline import train_baseline_from_frame
from pullback_prediction.training.benchmark import benchmark_tabular_models
from pullback_prediction.training.inference import predict_frame
from pullback_prediction.training.regime import RegimeDetector
from pullback_prediction.training.sequence_pipeline import train_sequence_from_frame
from pullback_prediction.training.stacking import train_stacked_from_frame
from pullback_prediction.training.walk_forward import walk_forward_evaluate

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


@app.command("walk-forward")
def walk_forward_command(
    input: Path = typer.Option(..., "--input", exists=True, readable=True),
    output_dir: Path = typer.Option(Path("artifacts/walk_forward"), "--output-dir"),
    n_splits: int = typer.Option(5, "--n-splits", min=2),
    horizon_bars: int = typer.Option(12, "--horizon-bars", min=1),
    barrier_atr_multiple: float = typer.Option(
        1.5, "--barrier-atr-multiple", min=0.01
    ),
    gap_bars: int | None = typer.Option(None, "--gap-bars", min=0),
    max_iter: int = typer.Option(200, "--max-iter", min=1),
) -> None:
    """Run expanding-window evaluation on 2015–2023; keep 2024+ untouched."""
    frame = pd.read_parquet(input)
    partitions = chronological_split(frame)
    development = pd.concat([partitions["train"], partitions["validation"]])
    if development.empty:
        raise typer.BadParameter("No development data exists before the final-test period.")
    effective_gap = horizon_bars if gap_bars is None else gap_bars
    if effective_gap < horizon_bars:
        raise typer.BadParameter(
            "gap-bars must be at least horizon-bars to prevent label leakage."
        )
    features = build_features(development)
    labels = make_barrier_labels(
        development,
        features["atr_14"],
        horizon_bars=horizon_bars,
        barrier_atr_multiple=barrier_atr_multiple,
    )
    result = walk_forward_evaluate(
        features,
        labels,
        n_splits=n_splits,
        gap_bars=effective_gap,
        max_iter=max_iter,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    folds_path = output_dir / "folds.csv"
    summary_path.write_text(
        json.dumps(
            result["summary"], indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(result["folds"]).to_csv(folds_path, index=False)
    typer.echo(f"summary_path={summary_path}")
    typer.echo(f"folds_path={folds_path}")
    for key, value in result["summary"].items():
        typer.echo(f"{key}={value}")


@app.command("benchmark-models")
def benchmark_models_command(
    input: Path = typer.Option(..., "--input", exists=True, readable=True),
    output_dir: Path = typer.Option(
        Path("artifacts/model_benchmark"), "--output-dir"
    ),
    n_splits: int = typer.Option(5, "--n-splits", min=2),
    horizon_bars: int = typer.Option(12, "--horizon-bars", min=1),
    barrier_atr_multiple: float = typer.Option(
        1.5, "--barrier-atr-multiple", min=0.01
    ),
    gap_bars: int | None = typer.Option(None, "--gap-bars", min=0),
    max_iter: int = typer.Option(200, "--max-iter", min=1),
    include_xgboost: bool = typer.Option(True, "--xgboost/--no-xgboost"),
) -> None:
    """Compare tabular models on expanding folds; never use 2024+ final test."""
    frame = pd.read_parquet(input)
    partitions = chronological_split(frame)
    development = pd.concat([partitions["train"], partitions["validation"]])
    if development.empty:
        raise typer.BadParameter(
            "No development data exists before final-test period."
        )
    effective_gap = horizon_bars if gap_bars is None else gap_bars
    if effective_gap < horizon_bars:
        raise typer.BadParameter("gap-bars must be at least horizon-bars.")
    features = build_features(development)
    labels = make_barrier_labels(
        development,
        features["atr_14"],
        horizon_bars=horizon_bars,
        barrier_atr_multiple=barrier_atr_multiple,
    )
    result = benchmark_tabular_models(
        features,
        labels,
        n_splits=n_splits,
        gap_bars=effective_gap,
        max_iter=max_iter,
        include_xgboost=include_xgboost,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(
            result["summary"], indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    for name, folds in result["folds"].items():
        pd.DataFrame(folds).to_csv(output_dir / f"{name}_folds.csv", index=False)
    (output_dir / "skipped.json").write_text(
        json.dumps(result["skipped"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    typer.echo(f"output_dir={output_dir}")
    for name, metrics in result["summary"].items():
        typer.echo(f"{name}={json.dumps(metrics, sort_keys=True)}")
    for name, reason in result["skipped"].items():
        typer.echo(f"skipped_{name}={reason}")


@app.command("fit-regimes")
def fit_regimes_command(
    input: Path = typer.Option(..., "--input", exists=True, readable=True),
    output_dir: Path = typer.Option(Path("artifacts/regimes"), "--output-dir"),
    n_regimes: int = typer.Option(4, "--n-regimes", min=2),
) -> None:
    """Fit market regimes on 2015–2020 only, then assign all available candles."""
    frame = pd.read_parquet(input)
    partitions = chronological_split(frame)
    training = partitions["train"]
    if training.empty:
        raise typer.BadParameter("Training partition 2015–2020 is empty.")
    features = build_features(frame)
    training_features = features.loc[training.index]
    detector = RegimeDetector.fit(training_features, n_regimes=n_regimes)
    assignments = detector.transform(features)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "regime_detector.joblib"
    assignments_path = output_dir / "regime_assignments.parquet"
    joblib.dump(
        {
            "detector": detector,
            "training_period": "2015-01-01 through 2020-12-31",
            "fit_rows": int(assignments.loc[training.index, "regime_id"].notna().sum()),
            "n_regimes": n_regimes,
        },
        model_path,
    )
    assignments.to_parquet(assignments_path)
    typer.echo(f"model_path={model_path}")
    typer.echo(f"assignments_path={assignments_path}")
    typer.echo(f"fit_rows={int(assignments.loc[training.index, 'regime_id'].notna().sum())}")
    typer.echo(f"assigned_rows={int(assignments['regime_id'].notna().sum())}")


@app.command("train-baseline")
def train_baseline(
    input: Path = typer.Option(..., "--input", exists=True, readable=True),
    output: Path = typer.Option(Path("artifacts/xauusd_baseline.joblib"), "--output"),
    horizon_bars: int = typer.Option(12, "--horizon-bars", min=1),
    barrier_atr_multiple: float = typer.Option(1.5, "--barrier-atr-multiple", min=0.01),
    max_iter: int = typer.Option(200, "--max-iter", min=1),
    timeframe: str = typer.Option("M1", "--timeframe"),
) -> None:
    """Train a gradient-boosting baseline; final-test data is excluded."""
    frame = pd.read_parquet(input)
    result = train_baseline_from_frame(
        frame,
        output,
        horizon_bars=horizon_bars,
        barrier_atr_multiple=barrier_atr_multiple,
        max_iter=max_iter,
        timeframe=timeframe,
    )
    for key, value in result.items():
        typer.echo(f"{key}={value}")


@app.command("train-stacked")
def train_stacked_command(
    input: Path = typer.Option(..., "--input", exists=True, readable=True),
    output: Path = typer.Option(Path("artifacts/xauusd_stacked.joblib"), "--output"),
    timeframe: str = typer.Option("M1", "--timeframe"),
    horizon_bars: int = typer.Option(12, "--horizon-bars", min=1),
    barrier_atr_multiple: float = typer.Option(
        1.5, "--barrier-atr-multiple", min=0.01
    ),
    n_splits: int = typer.Option(5, "--n-splits", min=2),
    gap_bars: int | None = typer.Option(None, "--gap-bars", min=0),
    max_iter: int = typer.Option(200, "--max-iter", min=1),
) -> None:
    """Train an OOF stack, calibrate 2021–2022 and evaluate 2023."""
    frame = pd.read_parquet(input)
    result = train_stacked_from_frame(
        frame,
        str(output),
        timeframe=timeframe,
        horizon_bars=horizon_bars,
        barrier_atr_multiple=barrier_atr_multiple,
        n_splits=n_splits,
        gap_bars=gap_bars,
        max_iter=max_iter,
    )
    for key, value in result.items():
        typer.echo(f"{key}={value}")


@app.command("train-sequence")
def train_sequence_command(
    input: Path = typer.Option(..., "--input", exists=True, readable=True),
    output: Path = typer.Option(Path("artifacts/xauusd_sequence.joblib"), "--output"),
    architecture: str = typer.Option("lstm", "--architecture"),
    timeframe: str = typer.Option("M1", "--timeframe"),
    sequence_length: int = typer.Option(128, "--sequence-length", min=2),
    horizon_bars: int = typer.Option(12, "--horizon-bars", min=1),
    barrier_atr_multiple: float = typer.Option(1.5, "--barrier-atr-multiple", min=0.01),
    hidden_dim: int = typer.Option(96, "--hidden-dim", min=8),
    num_layers: int = typer.Option(2, "--num-layers", min=1),
    dropout: float = typer.Option(0.15, "--dropout", min=0.0, max=0.9),
    epochs: int = typer.Option(30, "--epochs", min=1),
    batch_size: int = typer.Option(512, "--batch-size", min=1),
    learning_rate: float = typer.Option(1e-3, "--learning-rate", min=0.0),
    patience: int = typer.Option(5, "--patience", min=1),
    device: str | None = typer.Option(None, "--device"),
    mixed_precision: bool = typer.Option(True, "--mixed-precision/--no-mixed-precision"),
    num_workers: int = typer.Option(0, "--num-workers", min=0),
) -> None:
    """Train LSTM/Transformer on 2015–2020; calibrate 2021–2022; evaluate 2023."""
    frame = pd.read_parquet(input)
    result = train_sequence_from_frame(
        frame, output, architecture=architecture, timeframe=timeframe,
        sequence_length=sequence_length, horizon_bars=horizon_bars,
        barrier_atr_multiple=barrier_atr_multiple, hidden_dim=hidden_dim,
        num_layers=num_layers, dropout=dropout, epochs=epochs,
        batch_size=batch_size, learning_rate=learning_rate, patience=patience,
        device=device, mixed_precision=mixed_precision, num_workers=num_workers,
    )
    for key, value in result.items():
        typer.echo(f"{key}={value}")


@app.command("predict")
def predict_command(
    input: Path = typer.Option(..., "--input", exists=True, readable=True),
    model: Path = typer.Option(..., "--model", exists=True, readable=True),
    output: Path = typer.Option(Path("data/processed/predictions.parquet"), "--output"),
    min_probability: float = typer.Option(0.60, "--min-probability", min=0.0, max=1.0),
    timeframe: str = typer.Option("M1", "--timeframe"),
) -> None:
    """Generate BUY/SELL/WAIT signals and class scores from a saved model."""
    frame = pd.read_parquet(input)
    predictions = predict_frame(
        frame, model, min_probability=min_probability, timeframe=timeframe
    )
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
