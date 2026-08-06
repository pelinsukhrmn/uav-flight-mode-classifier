# ArduPilot .bin dataflash log okuma ve arıza ground-truth çıkarımı.
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymavlink import mavutil

LOG_FEATURES = ["vertical_speed", "horizontal_speed", "roll_angle", "pitch_angle",
                "motor_spread", "ekf_vel_innov", "baro_climb_rate",
                "roll_track_err", "pitch_track_err"]

EV_ARMED_IDS = (10, 15)
ERR_CODE_RESOLVED = 0

ERR_SUBSYS_TO_FAULT = {
    25: "motor_out",
    11: "gps_glitch",
    17: "gps_glitch",
    18: "sensor_freeze",
}


def load_flight_log(log_path):
    log = mavutil.mavlink_connection(str(log_path))

    att_rows, ctun_rows, gps_rows, rcou_rows, xkf_rows, baro_rows = [], [], [], [], [], []
    while True:
        msg = log.recv_match(type=["ATT", "CTUN", "GPS", "RCOU", "XKF4", "BARO"], blocking=False)
        if msg is None:
            break
        msg_type = msg.get_type()
        if msg_type == "ATT":
            att_rows.append({
                "timestamp": msg.TimeUS, "roll_angle": msg.Roll, "pitch_angle": msg.Pitch,
                "roll_track_err": abs(msg.DesRoll - msg.Roll), "pitch_track_err": abs(msg.DesPitch - msg.Pitch),
            })
        elif msg_type == "CTUN":
            ctun_rows.append({"timestamp": msg.TimeUS, "vertical_speed": msg.CRt})
        elif msg_type == "GPS":
            gps_rows.append({"timestamp": msg.TimeUS, "horizontal_speed": msg.Spd})
        elif msg_type == "RCOU":
            outputs = [msg.C1, msg.C2, msg.C3, msg.C4]
            rcou_rows.append({"timestamp": msg.TimeUS, "motor_spread": max(outputs) - min(outputs)})
        elif msg_type == "XKF4" and getattr(msg, "C", 0) == 0:
            xkf_rows.append({"timestamp": msg.TimeUS, "ekf_vel_innov": msg.SV})
        elif msg_type == "BARO" and getattr(msg, "I", 0) == 0:
            baro_rows.append({"timestamp": msg.TimeUS, "baro_alt": msg.Alt})

    def to_frame(rows):
        frame = pd.DataFrame(rows)
        return frame.sort_values("timestamp") if not frame.empty else frame

    att_df = to_frame(att_rows)
    ctun_df = to_frame(ctun_rows)
    gps_df = to_frame(gps_rows)
    rcou_df = to_frame(rcou_rows)
    xkf_df = to_frame(xkf_rows)
    baro_df = to_frame(baro_rows)

    if not baro_df.empty:
        baro_df["baro_climb_rate"] = (
            baro_df["baro_alt"].diff() / (baro_df["timestamp"].diff() / 1e6)
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        baro_df = baro_df[["timestamp", "baro_climb_rate"]]

    data = att_df
    for frame, column in ((ctun_df, "vertical_speed"), (gps_df, "horizontal_speed"),
                          (rcou_df, "motor_spread"), (xkf_df, "ekf_vel_innov"),
                          (baro_df, "baro_climb_rate")):
        if data.empty:
            break
        if frame.empty:
            data[column] = 0.0
        else:
            data = pd.merge_asof(data, frame, on="timestamp", direction="nearest")
    if data.empty:
        return data
    return data[["timestamp"] + LOG_FEATURES]


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
        if msg.get_type() == "EV" and armed_time_us is None and getattr(msg, "Id", None) in EV_ARMED_IDS:
            armed_time_us = msg.TimeUS
        elif msg.get_type() == "ERR":
            err_rows.append(msg)

    if not err_rows:
        return None

    data = load_flight_log(log_path)
    if data.empty:
        return None
    if armed_time_us is None:
        armed_time_us = data["timestamp"].iloc[0]

    ground_truth_mode = pd.Series("normal", index=data.index)
    any_fault_mapped = False
    open_faults = {}
    for msg in err_rows:
        if msg.TimeUS < armed_time_us:
            continue
        subsys = getattr(msg, "Subsys", None)
        fault = ERR_SUBSYS_TO_FAULT.get(subsys)
        if fault is None:
            continue
        if getattr(msg, "ECode", None) == ERR_CODE_RESOLVED:
            start_us = open_faults.pop(subsys, None)
            if start_us is not None:
                in_window = (data["timestamp"] >= start_us) & (data["timestamp"] < msg.TimeUS)
                ground_truth_mode[in_window] = fault
                any_fault_mapped = True
            continue
        open_faults.setdefault(subsys, msg.TimeUS)

    for subsys, start_us in open_faults.items():
        ground_truth_mode[data["timestamp"] >= start_us] = ERR_SUBSYS_TO_FAULT[subsys]
        any_fault_mapped = True

    if not any_fault_mapped:
        return None
    return pd.DataFrame({"timestamp": data["timestamp"], "ground_truth_mode": ground_truth_mode}).sort_values("timestamp")


def load_fault_ground_truth(log_path):
    scripted = load_scripted_fault_ground_truth(log_path)
    if scripted is not None:
        return scripted
    return load_native_fault_ground_truth(log_path)
