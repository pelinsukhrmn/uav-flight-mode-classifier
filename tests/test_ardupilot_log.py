# ardupilot_log.py'nin arıza ground-truth çıkarım fonksiyonlarını sahte veri/log ile test eder.
import json
from types import SimpleNamespace

import pandas as pd

import ardupilot_log as al


def test_load_scripted_fault_ground_truth_labels_windows_from_sidecar(tmp_path, monkeypatch):
    log_path = tmp_path / "sitl_motor_out_1.bin"
    log_path.write_bytes(b"")
    sidecar_path = tmp_path / "sitl_motor_out_1.fault_windows.json"
    sidecar_path.write_text(json.dumps([{"fault": "motor_out", "start_s": 2.0, "end_s": 4.0}]))

    fake_data = pd.DataFrame({
        "timestamp": [0, 1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000],
        "vertical_speed": [0.0] * 6,
    })
    monkeypatch.setattr(al, "load_flight_log", lambda path: fake_data.copy())

    ground_truth_df = al.load_scripted_fault_ground_truth(log_path)

    assert list(ground_truth_df["ground_truth_mode"]) == ["normal", "normal", "motor_out", "motor_out", "normal", "normal"]


def test_load_scripted_fault_ground_truth_returns_none_without_sidecar(tmp_path):
    log_path = tmp_path / "sitl_motor_out_2.bin"
    log_path.write_bytes(b"")
    assert al.load_scripted_fault_ground_truth(log_path) is None


def _fake_message(msg_type, **fields):
    return SimpleNamespace(get_type=lambda: msg_type, **fields)


class FakeConnection:
    def __init__(self, messages):
        self._messages = list(messages)

    def recv_match(self, type=None, blocking=False):
        if not self._messages:
            return None
        return self._messages.pop(0)


def test_load_native_fault_ground_truth_maps_thrust_loss_check_after_arming(monkeypatch):
    messages = [
        _fake_message("EV", Id=al.EV_ARMED_ID, TimeUS=1_000_000),
        _fake_message("ERR", Subsys=25, ECode=1, TimeUS=3_000_000),
    ]
    monkeypatch.setattr(al.mavutil, "mavlink_connection", lambda path: FakeConnection(messages))

    fake_data = pd.DataFrame({
        "timestamp": [0, 1_000_000, 2_000_000, 3_000_000, 4_000_000],
        "vertical_speed": [0.0] * 5,
    })
    monkeypatch.setattr(al, "load_flight_log", lambda path: fake_data.copy())

    ground_truth_df = al.load_native_fault_ground_truth("fake_log_path")

    assert list(ground_truth_df["ground_truth_mode"]) == ["normal", "normal", "normal", "motor_out", "motor_out"]


def test_load_native_fault_ground_truth_returns_none_without_armed_event(monkeypatch):
    messages = [_fake_message("ERR", Subsys=25, ECode=1, TimeUS=3_000_000)]
    monkeypatch.setattr(al.mavutil, "mavlink_connection", lambda path: FakeConnection(messages))
    assert al.load_native_fault_ground_truth("fake_log_path") is None


def test_load_fault_ground_truth_prefers_sidecar_over_native(tmp_path, monkeypatch):
    log_path = tmp_path / "sitl_gps_glitch_1.bin"
    log_path.write_bytes(b"")
    sidecar_path = tmp_path / "sitl_gps_glitch_1.fault_windows.json"
    sidecar_path.write_text(json.dumps([{"fault": "gps_glitch", "start_s": 0.0, "end_s": 1.0}]))

    fake_data = pd.DataFrame({"timestamp": [0, 1_000_000], "vertical_speed": [0.0, 0.0]})
    monkeypatch.setattr(al, "load_flight_log", lambda path: fake_data.copy())
    monkeypatch.setattr(al, "load_native_fault_ground_truth", lambda path: (_ for _ in ()).throw(AssertionError("should not be called")))

    ground_truth_df = al.load_fault_ground_truth(log_path)
    assert list(ground_truth_df["ground_truth_mode"]) == ["gps_glitch", "normal"]
