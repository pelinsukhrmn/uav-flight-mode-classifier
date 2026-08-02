from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from flight_mode_inference import (
    load_flight_log, load_artifacts, predict, summarize_segments, build_timeline_figure,
    load_ground_truth, attach_ground_truth, evaluate_predictions,
)

st.set_page_config(page_title="UAV Flight Mode Classifier", layout="wide")
st.title("UAV Flight Mode Classifier")
st.caption("Runs the trained sequence model on a PX4 flight log and shows the predicted mode over time.")

ARTIFACT_FILES = ["flight_mode_model.keras", "flight_mode_scaler.joblib", "flight_mode_meta.joblib"]
if not all(Path(f).exists() for f in ARTIFACT_FILES):
    st.error("No trained model found. Run `python flight_sequence_classifier.py` first to train and save one.")
    st.stop()

BUNDLED_LOGS = {
    "Sample log (bench/attitude test)": "data/sample.ulg",
    "Real flight 1 (slow, persistent pitch trim)": "data/real_flight.ulg",
    "Real flight 2 (fast, near-zero trim)": "data/real_flight_2.ulg",
    "Real flight 3 (VTOL, mostly manual)": "data/real_flight_3_vtol.ulg",
    "Real flight 4 (position hold, manual)": "data/real_flight_4_poshold.ulg",
    "Real flight 5 (stabilized, manual)": "data/real_flight_5_stab.ulg",
    "Real flight 6 (public log, takeoff/land/hover)": "data/real_flight_6_takeoff_land.ulg",
    "Real flight 7 (public log, takeoff/land)": "data/real_flight_7_takeoff_land.ulg",
    "Real flight 8 (public log, hover/rtl)": "data/real_flight_8_hover_rtl.ulg",
    "Real flight 9 (public log, hover/land)": "data/real_flight_9_hover_land.ulg",
}
BUNDLED_LOGS = {label: path for label, path in BUNDLED_LOGS.items() if Path(path).exists()}

uploaded = st.file_uploader("Upload a PX4 .ulg flight log", type=["ulg"])

log_path = None
log_label = None
if uploaded is not None:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ulg")
    tmp.write(uploaded.getvalue())
    tmp.close()
    log_path = tmp.name
    log_label = uploaded.name
elif BUNDLED_LOGS:
    choice = st.selectbox("...or pick a bundled log", list(BUNDLED_LOGS))
    log_path = BUNDLED_LOGS[choice]
    log_label = log_path

if log_path is None:
    st.warning("Upload a .ulg file to get started.")
    st.stop()

with st.spinner("Loading log and running inference..."):
    data = load_flight_log(log_path)
    ground_truth_df = load_ground_truth(log_path)
    if ground_truth_df is not None:
        data = attach_ground_truth(data, ground_truth_df)
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

evaluation = evaluate_predictions(data, result) if ground_truth_df is not None else None
if evaluation is not None:
    st.metric(
        "Ground-truth accuracy (vs PX4's own nav_state)",
        f"{evaluation['accuracy'] * 100:.1f}%",
        help=f"Covers {evaluation['coverage'] * 100:.1f}% of the flight ({evaluation['n_evaluated']} windows) "
             "- only nav_states with an unambiguous match to one of our labels (hover/takeoff/land/rtl) count.",
    )

st.subheader("Predicted mode distribution")
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

st.subheader("Next-mode forecast")
NEXT_ARTIFACT_FILES = ["flight_mode_next_model.keras", "flight_mode_next_scaler.joblib", "flight_mode_next_meta.joblib"]
if not all(Path(f).exists() for f in NEXT_ARTIFACT_FILES):
    st.info("No next-mode forecaster found. Run `python flight_sequence_classifier.py` to train one.")
else:
    next_meta, next_scaler, next_model = load_artifacts("flight_mode_next")
    next_result = predict(data, next_meta, next_scaler, next_model)
    if next_result is None:
        st.warning("Not enough samples in this log to forecast the next mode.")
    else:
        horizon = next_meta["horizon"]
        avg_dt = duration_seconds / max(len(data) - 1, 1)
        ncol1, ncol2 = st.columns(2)
        ncol1.metric(
            f"Predicted next mode (~{horizon * avg_dt:.1f}s ahead)",
            next_result["predicted_labels"][-1],
            help=f"Forecast made {horizon} steps (~{horizon * avg_dt:.1f}s at this log's sample rate) ahead of the window it's based on.",
        )
        ncol2.metric("Forecast confidence", f"{next_result['confidences'][-1]:.3f}")

        next_evaluation = evaluate_predictions(data, next_result, horizon=horizon) if ground_truth_df is not None else None
        if next_evaluation is not None:
            st.metric(
                "Next-mode ground-truth accuracy",
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
