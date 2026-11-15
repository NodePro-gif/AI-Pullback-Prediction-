# Kaggle GPU workflow

1. Open `kaggle/AI_Pullback_Prediction_Kaggle.ipynb` from this repository in Kaggle (or upload the notebook).
2. In Notebook **Settings → Accelerator**, select **GPU**.
3. Use **Add Input** to attach a prepared XAUUSD Parquet dataset created by this project's `prepare-data` command. The Parquet must contain bid/ask OHLC columns and a UTC DatetimeIndex.
4. Run all cells. If more than one Parquet is attached, set `DATA_PATH` explicitly in the configuration cell.
5. Results are written to `/kaggle/working/artifacts/`: model metadata `.joblib`, weights `.pt`, and metrics `.metrics.json`.

The notebook clones this repository and installs its `deep-learning` extra. Kaggle Internet must be enabled for cloning and package installation. GPU mode is required by the notebook and CUDA availability is checked before training.

Default settings are conservative for common Kaggle GPUs (batch size 256, sequence length 128). If CUDA runs out of memory, reduce batch size (128, 64, 32). If memory allows, benchmark larger batches rather than assuming the largest batch is fastest.

The sequence pipeline uses 2015–2020 for training, 2021 for early stopping, 2022 for calibration, and 2023 for evaluation. It does not access 2024+ final-test rows. M1 and M5 require separate datasets and separate model artifacts.

This notebook does not download market data automatically. Attach a prepared dataset as a Kaggle Dataset first; raw bid/ask CSV exports should be validated and combined with `prepare-data` before training.
