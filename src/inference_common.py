# Otopilottan bağımsız, paylaşılan inference/değerlendirme boru hattı.
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras


def temporal_split(X, y, train_fraction=0.7):
    split = int(len(X) * train_fraction)
    return X[:split], y[:split], X[split:], y[split:]


def load_artifacts(prefix="fault"):
    meta = joblib.load(f"models/{prefix}_meta.joblib")
    scaler = joblib.load(f"models/{prefix}_scaler.joblib")
    model = keras.models.load_model(f"models/{prefix}_model.keras")
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
                "fault": labels[start_idx],
                "start_s": times[start_idx],
                "end_s": times[i - 1],
                "duration_s": times[i - 1] - times[start_idx],
                "mean_confidence": np.mean(confidences[start_idx:i]),
            })
            start_idx = i
    return segments


def build_timeline_figure(data, result, log_label=""):
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    window_end_idx = result["window_end_idx"]
    window_times = result["window_times"]
    classes = result["classes"]
    segments = summarize_segments(window_times, result["predicted_labels"], result["confidences"])

    fig, (speed_ax, angle_ax, fault_ax) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    speed_ax.plot(window_times, data["vertical_speed"].to_numpy()[window_end_idx], label="vertical_speed")
    speed_ax.plot(window_times, data["horizontal_speed"].to_numpy()[window_end_idx], label="horizontal_speed")
    speed_ax.set_ylabel("speed (m/s)")
    speed_ax.legend(loc="upper right")
    speed_ax.set_title(f"Fault-precursor timeline — {log_label}")

    angle_ax.plot(window_times, data["roll_angle"].to_numpy()[window_end_idx], label="roll_angle")
    angle_ax.plot(window_times, data["pitch_angle"].to_numpy()[window_end_idx], label="pitch_angle")
    angle_ax.set_ylabel("angle (deg)")
    angle_ax.legend(loc="upper right")

    color_map = {fault: cm.tab10(i % 10) for i, fault in enumerate(classes)}
    for seg in segments:
        fault_ax.fill_between([seg["start_s"], seg["end_s"]], 0, 1, color=color_map[seg["fault"]])
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[fault]) for fault in classes]
    fault_ax.legend(handles, classes, loc="upper right", ncol=len(classes) // 2 or 1, fontsize="small")
    fault_ax.set_yticks([])
    fault_ax.set_xlabel("time (s)")
    fault_ax.set_ylabel("predicted class")

    fig.tight_layout()
    return fig


def attach_ground_truth(data, ground_truth_df):
    return pd.merge_asof(data, ground_truth_df, on="timestamp", direction="nearest")


def evaluate_predictions(data_with_gt, result, horizon=0):
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


def extract_labeled_real_windows(log_path, meta, log_loader, ground_truth_loader, horizon=0):
    ground_truth_df = ground_truth_loader(log_path)
    if ground_truth_df is None:
        return None, None

    data = log_loader(log_path)
    data = attach_ground_truth(data, ground_truth_df)

    windows, window_end_idx = build_windows(data, meta)
    if windows is None:
        return None, None

    shifted_idx = window_end_idx + horizon
    in_bounds = shifted_idx < len(data)
    windows, shifted_idx = windows[in_bounds], shifted_idx[in_bounds]
    if len(shifted_idx) == 0:
        return None, None

    ground_truth = data["ground_truth_mode"].to_numpy()[shifted_idx]
    covered = ~pd.isna(ground_truth)
    if not covered.any():
        return None, None
    return windows[covered], ground_truth[covered]
