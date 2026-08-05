# ArduPilot .bin dataflash log okuma ve arıza ground-truth çıkarımı.
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymavlink import mavutil

EV_ARMED_ID = 10

ERR_SUBSYS_TO_FAULT = {
    25: "motor_out",
    11: "gps_glitch",
    17: "gps_glitch",
    18: "sensor_freeze",
}


def load_flight_log(log_path):
    log = mavutil.mavlink_connection(str(log_path))

    att_rows, ctun_rows, gps_rows = [], [], []
    while True:
        msg = log.recv_match(type=["ATT", "CTUN", "GPS"], blocking=False)
        if msg is None:
            break
        msg_type = msg.get_type()
        if msg_type == "ATT":
            att_rows.append({"timestamp": msg.TimeUS, "roll_angle": np.degrees(msg.Roll), "pitch_angle": np.degrees(msg.Pitch)})
        elif msg_type == "CTUN":
            ctun_rows.append({"timestamp": msg.TimeUS, "vertical_speed": msg.CRt})
        elif msg_type == "GPS":
            gps_rows.append({"timestamp": msg.TimeUS, "horizontal_speed": msg.Spd})

    att_df = pd.DataFrame(att_rows).sort_values("timestamp")
    ctun_df = pd.DataFrame(ctun_rows).sort_values("timestamp")
    gps_df = pd.DataFrame(gps_rows).sort_values("timestamp")

    data = pd.merge_asof(att_df, ctun_df, on="timestamp", direction="nearest")
    data = pd.merge_asof(data, gps_df, on="timestamp", direction="nearest")
    return data[["timestamp", "vertical_speed", "horizontal_speed", "roll_angle", "pitch_angle"]]


def _sidecar_path(log_path):
    return Path(log_path).with_name(Path(log_path).stem + ".fault_windows.json")


def load_scripted_fault_ground_truth(log_path):
    sidecar_path = _sidecar_path(log_path)
    if not sidecar_path.exists():
        return None

    fault_windows = json.loads(sidecar_path.read_text())
    data = load_flight_log(log_path)
    if data.empty:
        return None

    elapsed_s = (data["timestamp"] - data["timestamp"].iloc[0]) / 1e6
    ground_truth_mode = pd.Series("normal", index=data.index)
    for window in fault_windows:
        in_window = (elapsed_s >= window["start_s"]) & (elapsed_s < window["end_s"])
        ground_truth_mode[in_window] = window["fault"]

    return pd.DataFrame({"timestamp": data["timestamp"], "ground_truth_mode": ground_truth_mode}).sort_values("timestamp")


def load_native_fault_ground_truth(log_path):
    log = mavutil.mavlink_connection(str(log_path))

    armed_time_us = None
    err_rows = []
    while True:
        msg = log.recv_match(type=["EV", "ERR"], blocking=False)
        if msg is None:
            break
        if msg.get_type() == "EV" and armed_time_us is None and getattr(msg, "Id", None) == EV_ARMED_ID:
            armed_time_us = msg.TimeUS
        elif msg.get_type() == "ERR":
            err_rows.append(msg)

    if armed_time_us is None or not err_rows:
        return None

    data = load_flight_log(log_path)
    ground_truth_mode = pd.Series("normal", index=data.index)
    any_fault_mapped = False
    for msg in err_rows:
        if msg.TimeUS < armed_time_us:
            continue
        fault = ERR_SUBSYS_TO_FAULT.get(getattr(msg, "Subsys", None))
        if fault is None:
            continue
        any_fault_mapped = True
        ground_truth_mode[data["timestamp"] >= msg.TimeUS] = fault

    if not any_fault_mapped:
        return None
    return pd.DataFrame({"timestamp": data["timestamp"], "ground_truth_mode": ground_truth_mode}).sort_values("timestamp")


def load_fault_ground_truth(log_path):
    scripted = load_scripted_fault_ground_truth(log_path)
    if scripted is not None:
        return scripted
    return load_native_fault_ground_truth(log_path)
