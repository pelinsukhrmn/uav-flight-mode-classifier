import numpy as np
import pandas as pd
import pytest

from flight_mode_inference import (
    quaternion_to_roll_pitch, map_nav_state, attach_ground_truth, evaluate_predictions,
    build_windows, summarize_segments,
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
