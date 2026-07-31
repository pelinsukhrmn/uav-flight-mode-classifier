import numpy as np
import pandas as pd
import joblib
from pyulog import ULog
from tensorflow import keras
import matplotlib.cm as cm


# PX4 vehicle_status.nav_state values with an unambiguous match to one of our
# mission-cycle labels. Values left out on purpose (MANUAL, ALTCTL, POSCTL,
# STAB, ACRO, OFFBOARD, AUTO_MISSION, ...) don't map to a single label -
# AUTO_MISSION alone could be ascend/cruise/descend/hover depending on which
# leg of the mission it is, and the manual modes have no equivalent at all.
NAV_STATE_TO_MODE = {
    4: "hover",          # AUTO_LOITER
    17: "takeoff",       # AUTO_TAKEOFF
    22: "takeoff",       # AUTO_VTOL_TAKEOFF
    18: "land",          # AUTO_LAND
    20: "land",          # AUTO_PRECLAND
    5: "rtl",            # AUTO_RTL
}


def quaternion_to_roll_pitch(q0, q1, q2, q3):
    sinr_cosp = 2 * (q0 * q1 + q2 * q3)
    cosr_cosp = 1 - 2 * (q1 * q1 + q2 * q2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = np.clip(2 * (q0 * q2 - q3 * q1), -1.0, 1.0)
    pitch = np.arcsin(sinp)

    return np.degrees(roll), np.degrees(pitch)


def load_flight_log(log_path):
    log = ULog(log_path)
    local_position = log.get_dataset("vehicle_local_position")
    attitude = log.get_dataset("vehicle_attitude")

    position_df = pd.DataFrame({
        "timestamp": local_position.data["timestamp"],
        "vertical_speed": -local_position.data["vz"],
        "horizontal_speed": np.sqrt(local_position.data["vx"] ** 2 + local_position.data["vy"] ** 2),
    }).sort_values("timestamp")

    roll, pitch = quaternion_to_roll_pitch(
        attitude.data["q[0]"], attitude.data["q[1]"], attitude.data["q[2]"], attitude.data["q[3]"]
    )
    attitude_df = pd.DataFrame({
        "timestamp": attitude.data["timestamp"],
        "roll_angle": roll,
        "pitch_angle": pitch,
    }).sort_values("timestamp")

    return pd.merge_asof(position_df, attitude_df, on="timestamp", direction="nearest")


def map_nav_state(nav_state):
    """Map a Series of PX4 nav_state ints to our mode labels, None where ambiguous/unmapped."""
    return nav_state.map(NAV_STATE_TO_MODE)


def load_ground_truth(log_path):
    """Load PX4's own nav_state ground truth from a log, mapped to our labels.

    Returns a (timestamp, ground_truth_mode) DataFrame, or None if the log has
    no vehicle_status topic, or if none of its nav_states map to our labels
    (e.g. a bench test that never leaves MANUAL).
    """
    log = ULog(log_path)
    if "vehicle_status" not in [d.name for d in log.data_list]:
        return None

    status = log.get_dataset("vehicle_status")
    ground_truth_df = pd.DataFrame({
        "timestamp": status.data["timestamp"],
        "ground_truth_mode": map_nav_state(pd.Series(status.data["nav_state"])),
    }).sort_values("timestamp")

    if ground_truth_df["ground_truth_mode"].isna().all():
        return None
    return ground_truth_df


def attach_ground_truth(data, ground_truth_df):
    """Align a (timestamp, ground_truth_mode) frame onto `data` by nearest timestamp."""
    return pd.merge_asof(data, ground_truth_df, on="timestamp", direction="nearest")


def evaluate_predictions(data_with_gt, result, horizon=0):
    """Compare predictions to PX4's own nav_state ground truth, where available.

    Only windows landing on an unambiguously-mapped nav_state count towards
    accuracy - `coverage` reports how much of the flight that was. `horizon`
    shifts the comparison forward by that many samples, for a forecasting
    model whose prediction for a window is about a point past its end -
    windows whose shifted index runs past the end of the log are dropped.
    """
    if "ground_truth_mode" not in data_with_gt.columns:
        return None

    shifted_idx = result["window_end_idx"] + horizon
    in_bounds = shifted_idx < len(data_with_gt)
    shifted_idx = shifted_idx[in_bounds]
    predicted = np.array(result["predicted_labels"])[in_bounds]

    ground_truth = data_with_gt["ground_truth_mode"].to_numpy()[shifted_idx]
    covered = ~pd.isna(ground_truth)
    if not covered.any():
        return None

    accuracy = (predicted[covered] == ground_truth[covered]).mean()
    return {
        "accuracy": accuracy,
        "coverage": covered.mean(),
        "n_evaluated": int(covered.sum()),
    }


def extract_labeled_real_windows(log_path, meta):
    """Real, PX4-nav_state-labeled (window, label) pairs from a log, for fine-tuning.

    Only windows landing on an unambiguously-mapped nav_state are kept (same
    restriction as evaluate_predictions) - returns (None, None) if the log
    has no such coverage at all.
    """
    data = load_flight_log(log_path)
    ground_truth_df = load_ground_truth(log_path)
    if ground_truth_df is None:
        return None, None
    data = attach_ground_truth(data, ground_truth_df)

    windows, window_end_idx = build_windows(data, meta)
    if windows is None:
        return None, None

    ground_truth = data["ground_truth_mode"].to_numpy()[window_end_idx]
    covered = ~pd.isna(ground_truth)
    if not covered.any():
        return None, None
    return windows[covered], ground_truth[covered]


def temporal_split(X, y, train_fraction=0.7):
    """Split (already-chronological) windows into an early train slice and a
    later holdout slice, so a holdout accuracy can't leak from overlapping
    windows the way a random shuffle-split would."""
    split = int(len(X) * train_fraction)
    return X[:split], y[:split], X[split:], y[split:]


def load_artifacts(prefix="flight_mode"):
    meta = joblib.load(f"{prefix}_meta.joblib")
    scaler = joblib.load(f"{prefix}_scaler.joblib")
    model = keras.models.load_model(f"{prefix}_model.keras")
    return meta, scaler, model


def build_windows(data, meta):
    features = meta["features"]
    window_size = meta["window_size"]

    base_features = [f for f in features if not f.endswith("_delta")]
    for feature in base_features:
        delta_feature = f"{feature}_delta"
        if delta_feature in features:
            data[delta_feature] = data[feature].diff().fillna(0)

    X = data[features].to_numpy()
    n_windows = len(X) - window_size + 1
    if n_windows < 1:
        return None, None

    windows = np.array([X[start:start + window_size] for start in range(n_windows)])
    window_end_idx = np.arange(window_size - 1, window_size - 1 + n_windows)
    return windows, window_end_idx


def predict(data, meta, scaler, model):
    windows, window_end_idx = build_windows(data, meta)
    if windows is None:
        return None

    n, w, f = windows.shape
    windows_scaled = scaler.transform(windows.reshape(-1, f)).reshape(n, w, f)

    probabilities = model.predict(windows_scaled, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    classes = meta["classes"]
    predicted_labels = [classes[i] for i in predictions]
    confidences = probabilities.max(axis=1)

    window_timestamps = data["timestamp"].to_numpy()[window_end_idx]
    window_times = (window_timestamps - data["timestamp"].iloc[0]) / 1e6

    return {
        "predicted_labels": predicted_labels,
        "confidences": confidences,
        "window_times": window_times,
        "window_end_idx": window_end_idx,
        "classes": classes,
    }


def summarize_segments(times, labels, confidences):
    segments = []
    start_idx = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start_idx]:
            segments.append({
                "mode": labels[start_idx],
                "start_s": times[start_idx],
                "end_s": times[i - 1],
                "duration_s": times[i - 1] - times[start_idx],
                "mean_confidence": np.mean(confidences[start_idx:i]),
            })
            start_idx = i
    return segments


def build_timeline_figure(data, result, log_label=""):
    import matplotlib.pyplot as plt

    window_end_idx = result["window_end_idx"]
    window_times = result["window_times"]
    classes = result["classes"]
    segments = summarize_segments(window_times, result["predicted_labels"], result["confidences"])

    fig, (speed_ax, angle_ax, mode_ax) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    speed_ax.plot(window_times, data["vertical_speed"].to_numpy()[window_end_idx], label="vertical_speed")
    speed_ax.plot(window_times, data["horizontal_speed"].to_numpy()[window_end_idx], label="horizontal_speed")
    speed_ax.set_ylabel("speed (m/s)")
    speed_ax.legend(loc="upper right")
    speed_ax.set_title(f"Flight mode timeline — {log_label}")

    angle_ax.plot(window_times, data["roll_angle"].to_numpy()[window_end_idx], label="roll_angle")
    angle_ax.plot(window_times, data["pitch_angle"].to_numpy()[window_end_idx], label="pitch_angle")
    angle_ax.set_ylabel("angle (deg)")
    angle_ax.legend(loc="upper right")

    color_map = {mode: cm.tab10(i % 10) for i, mode in enumerate(classes)}
    for seg in segments:
        mode_ax.fill_between([seg["start_s"], seg["end_s"]], 0, 1, color=color_map[seg["mode"]])
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[mode]) for mode in classes]
    mode_ax.legend(handles, classes, loc="upper right", ncol=len(classes) // 2 or 1, fontsize="small")
    mode_ax.set_yticks([])
    mode_ax.set_xlabel("time (s)")
    mode_ax.set_ylabel("predicted mode")

    fig.tight_layout()
    return fig
