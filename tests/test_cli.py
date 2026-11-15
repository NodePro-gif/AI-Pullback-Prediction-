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



def test_walk_forward_cli_never_includes_final_test_partition(tmp_path, monkeypatch):
    import json

    from pullback_prediction import cli

    train_index = pd.date_range("2015-01-01", periods=2, freq="min", tz="UTC")
    validation_index = pd.date_range("2021-01-01", periods=2, freq="min", tz="UTC")
    test_index = pd.date_range("2024-01-01", periods=2, freq="min", tz="UTC")
    index = train_index.append(validation_index).append(test_index)
    source = tmp_path / "candles.parquet"
    pd.DataFrame({"placeholder": range(len(index))}, index=index).to_parquet(source)
    output_dir = tmp_path / "walk-forward"
    observed = {}

    def fake_build_features(development):
        observed["feature_index"] = development.index
        return pd.DataFrame(
            {"atr_14": 1.0, "feature": range(len(development))},
            index=development.index,
        )

    def fake_labels(development, atr, horizon_bars, barrier_atr_multiple):
        observed["label_index"] = development.index
        return pd.Series([-1, 1, -1, 1], index=development.index)

    def fake_walk_forward(features, labels, **kwargs):
        observed["walk_index"] = features.index
        observed["gap_bars"] = kwargs["gap_bars"]
        return {
            "summary": {"folds": 2, "test_rows": 2, "weighted_accuracy": 0.5},
            "folds": [{"fold": 1, "accuracy": 0.5}],
        }

    monkeypatch.setattr(cli, "build_features", fake_build_features)
    monkeypatch.setattr(cli, "make_barrier_labels", fake_labels)
    monkeypatch.setattr(cli, "walk_forward_evaluate", fake_walk_forward)

    result = CliRunner().invoke(
        app,
        [
            "walk-forward",
            "--input",
            str(source),
            "--output-dir",
            str(output_dir),
            "--n-splits",
            "2",
            "--horizon-bars",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert observed["feature_index"].max() < pd.Timestamp("2024-01-01", tz="UTC")
    assert observed["label_index"].equals(observed["walk_index"])
    assert observed["gap_bars"] == 2
    assert (output_dir / "folds.csv").exists()
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["folds"] == 2


def test_fit_regimes_cli_fits_only_training_partition(tmp_path, monkeypatch):
    import numpy as np
    import joblib

    from pullback_prediction import cli
    from pullback_prediction.training.regime import DEFAULT_REGIME_FEATURES

    train_index = pd.date_range("2015-01-01", periods=240, freq="min", tz="UTC")
    validation_index = pd.date_range("2021-01-01", periods=4, freq="min", tz="UTC")
    test_index = pd.date_range("2024-01-01", periods=4, freq="min", tz="UTC")
    index = train_index.append(validation_index).append(test_index)
    source = tmp_path / "candles.parquet"
    pd.DataFrame({"placeholder": range(len(index))}, index=index).to_parquet(source)

    rng = np.random.default_rng(12)
    values = rng.normal(size=(len(index), len(DEFAULT_REGIME_FEATURES)))
    values[:, 0] = np.exp(values[:, 0]) * 0.001
    values[:, 1] = np.abs(values[:, 1]) * 0.0001
    values[:, 2] = np.abs(values[:, 2]) * 0.0002
    features = pd.DataFrame(values, index=index, columns=DEFAULT_REGIME_FEATURES)
    features.iloc[:5] = np.nan
    monkeypatch.setattr(cli, "build_features", lambda frame: features)

    output_dir = tmp_path / "regimes"
    result = CliRunner().invoke(
        app,
        [
            "fit-regimes",
            "--input",
            str(source),
            "--output-dir",
            str(output_dir),
            "--n-regimes",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    artifact = joblib.load(output_dir / "regime_detector.joblib")
    assert artifact["n_regimes"] == 3
    assert artifact["fit_rows"] == 235
    assignments = pd.read_parquet(output_dir / "regime_assignments.parquet")
    assert assignments.index.equals(index)
    assert assignments.loc[test_index, "regime_id"].notna().all()


def test_train_stacked_cli_command_is_registered():
    result = CliRunner().invoke(app, ["train-stacked", "--help"])
    assert result.exit_code == 0
    assert "--n-splits" in result.output
    assert "--gap-bars" in result.output
