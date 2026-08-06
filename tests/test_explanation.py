# inference_common.explain_window'un kanit siralamasini sahte model/scaler ile test eder.
import numpy as np

from inference_common import base_feature_columns, explain_window, format_explanation


class FakeScaler:
    def __init__(self, mean):
        self.mean_ = np.asarray(mean, dtype=float)

    def transform(self, X):
        return np.asarray(X, dtype=float) - self.mean_


class FakeModel:
    def __init__(self, driving_column):
        self.driving_column = driving_column

    def predict(self, X, verbose=0):
        strength = float(np.abs(X[0][:, self.driving_column]).mean())
        hit = min(0.99, strength / 100.0)
        return np.array([[1.0 - hit, hit]])


META = {
    "classes": ["normal", "motor_out"],
    "features": ["roll_angle", "motor_spread", "roll_angle_delta", "motor_spread_delta"],
    "window_size": 3,
}


def test_base_feature_columns_groups_deltas_with_their_base():
    columns = base_feature_columns(META["features"])
    assert columns["roll_angle"] == [0, 2]
    assert columns["motor_spread"] == [1, 3]


def test_explain_window_ranks_the_driving_feature_first():
    window = np.zeros((3, 4))
    window[:, 1] = 90.0
    scaler = FakeScaler([0.0, 0.0, 0.0, 0.0])
    model = FakeModel(driving_column=1)

    explanation = explain_window(window, scaler, model, META)

    assert explanation["label"] == "motor_out"
    assert explanation["evidence"][0]["feature"] == "motor_spread"
    assert explanation["evidence"][0]["contribution"] > 0
    assert explanation["evidence"][0]["observed"] == 90.0


def test_format_explanation_mentions_label_and_evidence():
    explanation = {
        "label": "motor_out",
        "confidence": 0.94,
        "evidence": [{"feature": "motor_spread", "contribution": 0.5, "observed": 640.0, "typical": 3.0}],
    }
    text = format_explanation(explanation)
    assert "motor_out" in text and "motor_spread=640.00" in text and "tipik 3.00" in text
