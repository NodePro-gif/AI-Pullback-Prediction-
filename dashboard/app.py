"""Local Streamlit research dashboard for saved model artifacts."""

from pathlib import Path

import pandas as pd
import streamlit as st

from pullback_prediction.training.inference import predict_frame

st.set_page_config(page_title="XAUUSD Pullback Research", layout="wide")
st.title("XAUUSD Pullback Prediction")
st.caption("Research-only. No broker connection or order execution.")

model_path = st.text_input("Model artifact", "artifacts/xauusd_baseline.joblib")
uploaded = st.file_uploader("Upload bid/ask candle Parquet", type=["parquet"])
threshold = st.slider("Minimum calibrated confidence", 0.0, 1.0, 0.60, 0.01)

if uploaded is not None:
    candles = pd.read_parquet(uploaded)
    st.write(f"Rows: {len(candles):,} | Period: {candles.index.min()} → {candles.index.max()}")
    if st.button("Generate predictions", type="primary"):
        if not Path(model_path).is_file():
            st.error("Model artifact not found.")
        else:
            try:
                predictions = predict_frame(candles, model_path, min_probability=threshold)
                if not predictions["probabilities_calibrated"].all():
                    st.warning("This artifact returns raw model scores, not calibrated probabilities.")
                counts = predictions["signal"].value_counts()
                left, middle, right = st.columns(3)
                left.metric("BUY", int(counts.get(1, 0)))
                middle.metric("WAIT", int(counts.get(0, 0)))
                right.metric("SELL", int(counts.get(-1, 0)))
                st.line_chart(predictions[["prob_sell", "prob_wait", "prob_buy"]])
                st.dataframe(predictions.tail(500), use_container_width=True)
            except Exception as exc:
                st.exception(exc)
