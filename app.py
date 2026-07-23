from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from flight_mode_inference import load_flight_log, load_artifacts, predict, summarize_segments, build_timeline_figure

st.set_page_config(page_title="UAV Flight Mode Classifier", layout="wide")
st.title("UAV Flight Mode Classifier")
st.caption("Runs the trained sequence model on a PX4 flight log and shows the predicted mode over time.")

ARTIFACT_FILES = ["flight_mode_model.keras", "flight_mode_scaler.joblib", "flight_mode_meta.joblib"]
if not all(Path(f).exists() for f in ARTIFACT_FILES):
    st.error("No trained model found. Run `python flight_sequence_classifier.py` first to train and save one.")
    st.stop()

SAMPLE_LOG = "data/sample.ulg"
uploaded = st.file_uploader("Upload a PX4 .ulg flight log", type=["ulg"])

log_path = None
log_label = None
if uploaded is not None:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ulg")
    tmp.write(uploaded.getvalue())
    tmp.close()
    log_path = tmp.name
    log_label = uploaded.name
elif Path(SAMPLE_LOG).exists():
    st.info(f"No file uploaded — using the bundled sample log ({SAMPLE_LOG}).")
    log_path = SAMPLE_LOG
    log_label = SAMPLE_LOG

if log_path is None:
    st.warning("Upload a .ulg file to get started.")
    st.stop()

with st.spinner("Loading log and running inference..."):
    data = load_flight_log(log_path)
    meta, scaler, model = load_artifacts()
    result = predict(data, meta, scaler, model)

if result is None:
    st.warning("Not enough samples in this log to build a single window.")
    st.stop()

duration_seconds = (data["timestamp"].iloc[-1] - data["timestamp"].iloc[0]) / 1e6

col1, col2, col3 = st.columns(3)
col1.metric("Flight duration", f"{duration_seconds:.1f}s")
col2.metric("Samples", len(data))
col3.metric("Mean confidence", f"{result['confidences'].mean():.3f}")

st.subheader("Predicted mode distribution")
st.bar_chart(pd.Series(result["predicted_labels"]).value_counts())

st.subheader("Timeline")
fig = build_timeline_figure(data, result, log_label)
st.pyplot(fig)

st.subheader("Predicted segments")
segments = summarize_segments(result["window_times"], result["predicted_labels"], result["confidences"])
st.dataframe(pd.DataFrame(segments), use_container_width=True)
