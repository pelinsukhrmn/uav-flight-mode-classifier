# ArduPilot uçuş loglarında arıza-öngörü modelini gezinmek için Streamlit arayüzü.
from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from inference_common import load_artifacts, predict, summarize_segments, build_timeline_figure, evaluate_predictions, attach_ground_truth
from ardupilot_log import load_flight_log, load_fault_ground_truth

st.set_page_config(page_title="UAV Fault Precursor Detector", layout="wide")
st.title("UAV Fault Precursor Detector")
st.caption("Runs the trained sequence model on an ArduPilot flight log and shows the predicted fault precursor over time.")

ARTIFACT_FILES = ["models/fault_model.keras", "models/fault_scaler.joblib", "models/fault_meta.joblib"]
if not all(Path(f).exists() for f in ARTIFACT_FILES):
    st.error("No trained model found. Run `python src/fault_sequence_classifier.py` first to train and save one.")
    st.stop()

BUNDLED_LOGS = {
    "SITL motor_out 1": "data/sitl_motor_out_1.bin",
    "SITL gps_glitch 1": "data/sitl_gps_glitch_1.bin",
    "SITL wind_gust_upset 1": "data/sitl_wind_gust_upset_1.bin",
    "SITL sensor_freeze 1": "data/sitl_sensor_freeze_1.bin",
}
BUNDLED_LOGS = {label: path for label, path in BUNDLED_LOGS.items() if Path(path).exists()}

uploaded = st.file_uploader("Upload an ArduPilot .bin flight log", type=["bin"])

log_path = None
log_label = None
if uploaded is not None:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
    tmp.write(uploaded.getvalue())
    tmp.close()
    log_path = tmp.name
    log_label = uploaded.name
elif BUNDLED_LOGS:
    choice = st.selectbox("...or pick a bundled log", list(BUNDLED_LOGS))
    log_path = BUNDLED_LOGS[choice]
    log_label = log_path

if log_path is None:
    st.warning("Upload a .bin file to get started.")
    st.stop()

with st.spinner("Loading log and running inference..."):
    data = load_flight_log(log_path)
    ground_truth_df = load_fault_ground_truth(log_path)
    if ground_truth_df is not None:
        data = attach_ground_truth(data, ground_truth_df)
    meta, scaler, model = load_artifacts("fault")
    result = predict(data, meta, scaler, model)

if result is None:
    st.warning("Not enough samples in this log to build a single window.")
    st.stop()

duration_seconds = (data["timestamp"].iloc[-1] - data["timestamp"].iloc[0]) / 1e6

col1, col2, col3 = st.columns(3)
col1.metric("Flight duration", f"{duration_seconds:.1f}s")
col2.metric("Samples", len(data))
col3.metric("Mean confidence", f"{result['confidences'].mean():.3f}")

evaluation = evaluate_predictions(data, result) if ground_truth_df is not None else None
if evaluation is not None:
    st.metric(
        "Ground-truth accuracy (vs ArduPilot's ERR/MODE.Rsn or scripted SITL windows)",
        f"{evaluation['accuracy'] * 100:.1f}%",
        help=f"Covers {evaluation['coverage'] * 100:.1f}% of the flight ({evaluation['n_evaluated']} windows).",
    )

st.subheader("Predicted fault distribution")
st.bar_chart(pd.Series(result["predicted_labels"]).value_counts())

st.subheader("Timeline")
fig = build_timeline_figure(data, result, log_label)
st.pyplot(fig)

st.subheader("Predicted segments")
segments_df = pd.DataFrame(summarize_segments(result["window_times"], result["predicted_labels"], result["confidences"]))
st.dataframe(segments_df, use_container_width=True)
st.download_button(
    "Download segments as CSV",
    segments_df.to_csv(index=False),
    file_name=f"{Path(log_label).stem}_segments.csv",
    mime="text/csv",
)

st.subheader("Fault forecast")
NEXT_ARTIFACT_FILES = ["models/fault_next_model.keras", "models/fault_next_scaler.joblib", "models/fault_next_meta.joblib"]
if not all(Path(f).exists() for f in NEXT_ARTIFACT_FILES):
    st.info("No fault forecaster found. Run `python src/fault_sequence_classifier.py` to train one.")
else:
    next_meta, next_scaler, next_model = load_artifacts("fault_next")
    next_result = predict(data, next_meta, next_scaler, next_model)
    if next_result is None:
        st.warning("Not enough samples in this log to forecast the next fault.")
    else:
        horizon = next_meta["horizon"]
        avg_dt = duration_seconds / max(len(data) - 1, 1)
        ncol1, ncol2 = st.columns(2)
        ncol1.metric(
            f"Predicted fault (~{horizon * avg_dt:.1f}s ahead)",
            next_result["predicted_labels"][-1],
            help=f"Forecast made {horizon} steps (~{horizon * avg_dt:.1f}s at this log's sample rate) ahead of the window it's based on.",
        )
        ncol2.metric("Forecast confidence", f"{next_result['confidences'][-1]:.3f}")

        next_evaluation = evaluate_predictions(data, next_result, horizon=horizon) if ground_truth_df is not None else None
        if next_evaluation is not None:
            st.metric(
                "Forecast ground-truth accuracy",
                f"{next_evaluation['accuracy'] * 100:.1f}%",
                help=f"Covers {next_evaluation['coverage'] * 100:.1f}% of the flight ({next_evaluation['n_evaluated']} windows).",
            )

        forecast_segments_df = pd.DataFrame(
            summarize_segments(next_result["window_times"], next_result["predicted_labels"], next_result["confidences"])
        )
        st.dataframe(forecast_segments_df, use_container_width=True)
        st.download_button(
            "Download forecast segments as CSV",
            forecast_segments_df.to_csv(index=False),
            file_name=f"{Path(log_label).stem}_forecast_segments.csv",
            mime="text/csv",
        )
