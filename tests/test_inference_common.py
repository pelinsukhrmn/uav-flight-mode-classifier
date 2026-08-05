# inference_common.py'nin otopilottan bağımsız pencere/değerlendirme fonksiyonlarını test eder.
import numpy as np
import pandas as pd
import pytest

import inference_common as ic
from inference_common import attach_ground_truth, evaluate_predictions, build_windows, summarize_segments, temporal_split


def test_attach_and_evaluate_predictions_only_counts_covered_windows():
    data = pd.DataFrame({
        "timestamp": [0, 1, 2, 3, 4],
        "vertical_speed": [0.0] * 5,
    })
    ground_truth_df = pd.DataFrame({
        "timestamp": [0, 1, 2, 3, 4],
        "ground_truth_mode": ["normal", "normal", None, "motor_out", "motor_out"],
    })
    data_with_gt = attach_ground_truth(data, ground_truth_df)

    result = {
        "window_end_idx": np.array([0, 1, 2, 3, 4]),
        "predicted_labels": ["normal", "gps_glitch", "normal", "motor_out", "normal"],
    }

    evaluation = evaluate_predictions(data_with_gt, result)

    assert evaluation["n_evaluated"] == 4
    assert evaluation["coverage"] == 4 / 5
    assert evaluation["accuracy"] == pytest.approx(2 / 4)


def test_evaluate_predictions_returns_none_without_ground_truth_column():
    data = pd.DataFrame({"timestamp": [0, 1], "vertical_speed": [0.0, 0.0]})
    result = {"window_end_idx": np.array([0, 1]), "predicted_labels": ["normal", "normal"]}
    assert evaluate_predictions(data, result) is None


def test_build_windows_produces_expected_shapes_and_delta_features():
    data = pd.DataFrame({"a": [1.0, 2.0, 4.0, 7.0]})
    meta = {"features": ["a", "a_delta"], "window_size": 2}

    windows, window_end_idx = build_windows(data, meta)

    assert windows.shape == (3, 2, 2)
    assert list(window_end_idx) == [1, 2, 3]
    assert windows[0, 0, 1] == 0.0
    assert windows[0, 1, 1] == 1.0


def test_build_windows_returns_none_when_too_few_rows():
    data = pd.DataFrame({"a": [1.0]})
    meta = {"features": ["a"], "window_size": 3}
    windows, window_end_idx = build_windows(data, meta)
    assert windows is None
    assert window_end_idx is None


def test_evaluate_predictions_with_horizon_shifts_ground_truth_forward():
    data = pd.DataFrame({
        "timestamp": [0, 1, 2, 3, 4, 5],
        "ground_truth_mode": ["normal", "normal", "normal", "motor_out", "motor_out", "motor_out"],
    })
    result = {
        "window_end_idx": np.array([0, 1, 2]),
        "predicted_labels": ["normal", "motor_out", "motor_out"],
    }

    evaluation = evaluate_predictions(data, result, horizon=3)

    assert evaluation["n_evaluated"] == 3
    assert evaluation["accuracy"] == pytest.approx(2 / 3)


def test_evaluate_predictions_with_horizon_drops_out_of_bounds_windows():
    data = pd.DataFrame({
        "timestamp": [0, 1, 2, 3],
        "ground_truth_mode": ["normal", "normal", "motor_out", "motor_out"],
    })
    result = {
        "window_end_idx": np.array([0, 1, 2, 3]),
        "predicted_labels": ["normal", "normal", "normal", "normal"],
    }

    evaluation = evaluate_predictions(data, result, horizon=2)

    assert evaluation["n_evaluated"] == 2
    assert evaluation["accuracy"] == pytest.approx(0.0)


def test_temporal_split_keeps_chronological_order():
    X = np.arange(10).reshape(10, 1, 1)
    y = np.array([str(i) for i in range(10)])

    X_train, y_train, X_holdout, y_holdout = temporal_split(X, y, train_fraction=0.7)

    assert len(X_train) == 7 and len(X_holdout) == 3
    assert list(y_train) == [str(i) for i in range(7)]
    assert list(y_holdout) == [str(i) for i in range(7, 10)]


def test_load_artifacts_uses_prefix_for_filenames(monkeypatch):
    calls = []
    monkeypatch.setattr(ic.joblib, "load", lambda path: calls.append(("joblib", path)))
    monkeypatch.setattr(ic.keras.models, "load_model", lambda path: calls.append(("keras", path)))

    ic.load_artifacts(prefix="fault_next")

    assert calls == [
        ("joblib", "models/fault_next_meta.joblib"),
        ("joblib", "models/fault_next_scaler.joblib"),
        ("keras", "models/fault_next_model.keras"),
    ]


def test_summarize_segments_groups_consecutive_equal_labels():
    times = np.array([0.0, 1.0, 2.0, 3.0])
    labels = ["normal", "normal", "motor_out", "motor_out"]
    confidences = np.array([0.9, 0.8, 0.7, 0.6])

    segments = summarize_segments(times, labels, confidences)

    assert len(segments) == 2
    assert segments[0]["fault"] == "normal"
    assert segments[0]["start_s"] == 0.0
    assert segments[0]["end_s"] == 1.0
    assert segments[1]["fault"] == "motor_out"
    assert segments[1]["start_s"] == 2.0
    assert segments[1]["end_s"] == 3.0


def test_extract_labeled_real_windows_uses_injected_loaders():
    data = pd.DataFrame({
        "timestamp": [0, 1, 2, 3, 4],
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    ground_truth_df = pd.DataFrame({
        "timestamp": [0, 1, 2, 3, 4],
        "ground_truth_mode": ["normal", "normal", "motor_out", "motor_out", "motor_out"],
    })
    meta = {"features": ["a"], "window_size": 2}

    windows, labels = ic.extract_labeled_real_windows(
        "fake_log_path", meta,
        log_loader=lambda path: data.copy(),
        ground_truth_loader=lambda path: ground_truth_df.copy(),
    )

    assert windows.shape[0] == len(labels)
    assert list(labels) == ["normal", "motor_out", "motor_out", "motor_out"]


def test_extract_labeled_real_windows_returns_none_without_ground_truth():
    data = pd.DataFrame({"timestamp": [0, 1], "a": [1.0, 2.0]})
    meta = {"features": ["a"], "window_size": 2}

    result = ic.extract_labeled_real_windows(
        "fake_log_path", meta,
        log_loader=lambda path: data.copy(),
        ground_truth_loader=lambda path: None,
    )
    assert result == (None, None)
