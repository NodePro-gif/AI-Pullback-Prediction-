# Kaggle GPU workflow (automatic data download)

1. Open `kaggle/AI_Pullback_Prediction_Kaggle.ipynb` in Kaggle.
2. In Notebook **Settings → Accelerator**, select **GPU** and turn **Internet** on.
3. Run all cells. The notebook clones this repository, installs PyTorch, installs Node.js/npm if needed, downloads XAUUSD bid and ask candles directly from Dukascopy, validates and combines them into Parquet, then trains the selected LSTM/Transformer model.
4. Default download: 2015-01-01 through 2024-01-01 (exclusive). This provides 2015–2020 training, 2021 early stopping, 2022 calibration and 2023 evaluation. It deliberately does not download/use 2024+.
5. Outputs are written to `/kaggle/working/artifacts/`: `.joblib` metadata, `.pt` weights and `.metrics.json`.

Set `TIMEFRAME` to `M5` for a separate M5 run. Kaggle Internet is required for GitHub, npm and Dukascopy access. Historical M1 downloads are large and may exceed a single Kaggle session's time or disk limits; the downloader uses retry/cache, but Kaggle's ephemeral working storage is not permanent. Downloaded files can be saved as a Kaggle Dataset between sessions if needed.

Batch size defaults to 256. If CUDA runs out of memory, reduce it to 128, 64 or 32. Benchmark throughput rather than assuming the largest batch is fastest.

No manual dataset upload is required. The notebook fetches market data itself. Training and evaluation metrics are research measurements, not a guarantee of profitability.
