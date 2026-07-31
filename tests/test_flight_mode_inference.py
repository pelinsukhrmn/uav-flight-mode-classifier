import numpy as np
import pandas as pd
import pytest

import flight_mode_inference as fmi
from flight_mode_inference import (
    quaternion_to_roll_pitch, map_nav_state, attach_ground_truth, evaluate_predictions,
    build_windows, summarize_segments, temporal_split,
)


def test_quaternion_identity_is_level():
    roll, pitch = quaternion_to_roll_pitch(1.0, 0.0, 0.0, 0.0)
    assert roll == pytest.approx(0.0)
    assert pitch == pytest.approx(0.0)


def test_map_nav_state_known_and_unknown_values():
    nav_state = pd.Series([4, 17, 18, 5, 22, 20, 3, 0, 15])
    mapped = map_nav_state(nav_state)

    assert list(mapped[:6]) == ["hover", "takeoff", "land", "rtl", "takeoff", "land"]
    assert mapped[6:].isna().all()  # AUTO_MISSION, MANUAL, STAB have no unambiguous label


def test_attach_and_evaluate_predictions_only_counts_covered_windows():
    data = pd.DataFrame({
        "timestamp": [0, 1, 2, 3, 4],
        "vertical_speed": [0.0] * 5,
    })
    ground_truth_df = pd.DataFrame({
        "timestamp": [0, 1, 2, 3, 4],
        "ground_truth_mode": ["hover", "hover", None, "rtl", "rtl"],
    })
    data_with_gt = attach_ground_truth(data, ground_truth_df)

    result = {
        "window_end_idx": np.array([0, 1, 2, 3, 4]),
        "predicted_labels": ["hover", "cruise", "hover", "rtl", "hover"],
    }

    evaluation = evaluate_predictions(data_with_gt, result)

    assert evaluation["n_evaluated"] == 4  # index 2 (None ground truth) excluded
    assert evaluation["coverage"] == 4 / 5
    assert evaluation["accuracy"] == pytest.approx(2 / 4)  # indices 0,3 correct; 1,4 wrong


def test_evaluate_predictions_returns_none_without_ground_truth_column():
    data = pd.DataFrame({"timestamp": [0, 1], "vertical_speed": [0.0, 0.0]})
    result = {"window_end_idx": np.array([0, 1]), "predicted_labels": ["hover", "hover"]}
    assert evaluate_predictions(data, result) is None


def test_build_windows_produces_expected_shapes_and_delta_features():
    data = pd.DataFrame({"a": [1.0, 2.0, 4.0, 7.0]})
    meta = {"features": ["a", "a_delta"], "window_size": 2}

    windows, window_end_idx = build_windows(data, meta)

    assert windows.shape == (3, 2, 2)
    assert list(window_end_idx) == [1, 2, 3]
    # a_delta for the first window: [nan->0 filled, 2-1]
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
        "ground_truth_mode": ["hover", "hover", "hover", "rtl", "rtl", "rtl"],
    })
    result = {
        "window_end_idx": np.array([0, 1, 2]),
        "predicted_labels": ["hover", "rtl", "rtl"],  # forecasts for indices 3, 4, 5
    }

    evaluation = evaluate_predictions(data, result, horizon=3)

    assert evaluation["n_evaluated"] == 3
    assert evaluation["accuracy"] == pytest.approx(2 / 3)  # index 0's forecast ("hover") was wrong


def test_evaluate_predictions_with_horizon_drops_out_of_bounds_windows():
    data = pd.DataFrame({
        "timestamp": [0, 1, 2, 3],
        "ground_truth_mode": ["hover", "hover", "rtl", "rtl"],
    })
    result = {
        "window_end_idx": np.array([0, 1, 2, 3]),
        "predicted_labels": ["hover", "hover", "hover", "hover"],
    }

    # horizon=2 shifts indices to [2, 3, 4, 5] - the last two run past the end (len=4) and are dropped
    evaluation = evaluate_predictions(data, result, horizon=2)

    assert evaluation["n_evaluated"] == 2
    assert evaluation["accuracy"] == pytest.approx(0.0)  # both remaining forecasts ("hover") missed the actual "rtl"


def test_temporal_split_keeps_chronological_order():
    X = np.arange(10).reshape(10, 1, 1)
    y = np.array([str(i) for i in range(10)])

    X_train, y_train, X_holdout, y_holdout = temporal_split(X, y, train_fraction=0.7)

    assert len(X_train) == 7 and len(X_holdout) == 3
    assert list(y_train) == [str(i) for i in range(7)]
    assert list(y_holdout) == [str(i) for i in range(7, 10)]


def test_load_artifacts_uses_prefix_for_filenames(monkeypatch):
    calls = []
    monkeypatch.setattr(fmi.joblib, "load", lambda path: calls.append(("joblib", path)))
    monkeypatch.setattr(fmi.keras.models, "load_model", lambda path: calls.append(("keras", path)))

    fmi.load_artifacts(prefix="flight_mode_next")

    assert calls == [
        ("joblib", "flight_mode_next_meta.joblib"),
        ("joblib", "flight_mode_next_scaler.joblib"),
        ("keras", "flight_mode_next_model.keras"),
    ]


def test_summarize_segments_groups_consecutive_equal_labels():
    times = np.array([0.0, 1.0, 2.0, 3.0])
    labels = ["hover", "hover", "rtl", "rtl"]
    confidences = np.array([0.9, 0.8, 0.7, 0.6])

    segments = summarize_segments(times, labels, confidences)

    assert len(segments) == 2
    assert segments[0]["mode"] == "hover"
    assert segments[0]["start_s"] == 0.0
    assert segments[0]["end_s"] == 1.0
    assert segments[1]["mode"] == "rtl"
    assert segments[1]["start_s"] == 2.0
    assert segments[1]["end_s"] == 3.0
