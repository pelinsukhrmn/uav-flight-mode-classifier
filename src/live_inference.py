# Canlı MAVLink veya log replay üzerinden akan arıza-öngörü karar-destek döngüsü.
import argparse
import time
from collections import deque

import numpy as np
import pandas as pd

from inference_common import load_artifacts, build_windows
from ardupilot_log import load_flight_log, LOG_FEATURES


def replay_log_source(log_path, speed=20.0):
    data = load_flight_log(log_path)
    prev_ts = None
    for _, row in data.iterrows():
        if speed and prev_ts is not None:
            dt = (row["timestamp"] - prev_ts) / 1e6
            if dt > 0:
                time.sleep(dt / speed)
        prev_ts = row["timestamp"]
        yield {"timestamp": row["timestamp"], **{feature: row[feature] for feature in LOG_FEATURES}}


SEA_LEVEL_HPA = 1013.25


def pressure_to_altitude_m(pressure_hpa):
    return 44330.0 * (1.0 - (pressure_hpa / SEA_LEVEL_HPA) ** 0.1903)


def quaternion_to_roll_pitch_deg(q):
    w, x, y, z = q
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return np.degrees(roll), np.degrees(pitch)


class MavlinkFeatureState:
    def __init__(self):
        self.roll_angle = None
        self.pitch_angle = None
        self.target_roll = 0.0
        self.target_pitch = 0.0
        self.motor_spread = 0.0
        self.ekf_vel_innov = 0.0
        self.baro_climb_rate = 0.0
        self._baro_alt = None
        self._baro_time_s = None

    def update(self, msg):
        msg_type = msg.get_type()
        if msg_type == "ATTITUDE":
            self.roll_angle = np.degrees(msg.roll)
            self.pitch_angle = np.degrees(msg.pitch)
            return None
        if msg_type == "ATTITUDE_TARGET":
            self.target_roll, self.target_pitch = quaternion_to_roll_pitch_deg(msg.q)
            return None
        if msg_type == "SERVO_OUTPUT_RAW":
            outputs = [msg.servo1_raw, msg.servo2_raw, msg.servo3_raw, msg.servo4_raw]
            self.motor_spread = float(max(outputs) - min(outputs))
            return None
        if msg_type == "EKF_STATUS_REPORT":
            self.ekf_vel_innov = float(msg.velocity_variance)
            return None
        if msg_type == "SCALED_PRESSURE":
            altitude = pressure_to_altitude_m(msg.press_abs)
            now_s = msg.time_boot_ms / 1000.0
            if self._baro_alt is not None and now_s > self._baro_time_s:
                self.baro_climb_rate = (altitude - self._baro_alt) / (now_s - self._baro_time_s)
            self._baro_alt, self._baro_time_s = altitude, now_s
            return None
        if msg_type != "LOCAL_POSITION_NED" or self.roll_angle is None:
            return None

        return {
            "timestamp": msg.time_boot_ms * 1000,
            "vertical_speed": -msg.vz,
            "horizontal_speed": (msg.vx ** 2 + msg.vy ** 2) ** 0.5,
            "roll_angle": self.roll_angle,
            "pitch_angle": self.pitch_angle,
            "motor_spread": self.motor_spread,
            "ekf_vel_innov": self.ekf_vel_innov,
            "baro_climb_rate": self.baro_climb_rate,
            "roll_track_err": abs(self.target_roll - self.roll_angle),
            "pitch_track_err": abs(self.target_pitch - self.pitch_angle),
        }


def mavlink_source(connection_string="udp:127.0.0.1:14550"):
    from pymavlink import mavutil

    conn = mavutil.mavlink_connection(connection_string)
    conn.wait_heartbeat()
    conn.mav.request_data_stream_send(conn.target_system, conn.target_component, mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1)

    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
        mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE_TARGET, 100000, 0, 0, 0, 0, 0,
    )

    state = MavlinkFeatureState()
    while True:
        msg = conn.recv_match(blocking=True)
        if msg is None:
            continue
        row = state.update(msg)
        if row is not None:
            yield row


class LiveWindowBuffer:
    def __init__(self, meta):
        self.meta = meta
        self.rows = deque(maxlen=meta["window_size"] + 1)

    def push(self, row):
        self.rows.append(row)

    def latest_window(self):
        if len(self.rows) < self.meta["window_size"] + 1:
            return None, None
        data = pd.DataFrame(self.rows)
        windows, window_end_idx = build_windows(data, self.meta)
        if windows is None:
            return None, None
        return windows[-1:], data["timestamp"].to_numpy()[window_end_idx[-1]]


def score_window(window, scaler, model, meta):
    n, w, f = window.shape
    scaled = scaler.transform(window.reshape(-1, f)).reshape(n, w, f)
    probs = model.predict(scaled, verbose=0)[0]
    idx = int(np.argmax(probs))
    return meta["classes"][idx], float(probs[idx])


def advisory(current_label, current_conf, next_label, next_conf, horizon_seconds):
    if next_label == current_label:
        return None
    urgent = next_label != "normal" and next_conf >= 0.6
    level = "UYARI" if urgent else "bilgi"
    return (
        f"[{level}] su an: {current_label} ({current_conf:.2f}) -> "
        f"~{horizon_seconds:.1f}s sonra tahmin: {next_label} ({next_conf:.2f})"
    )


def run(source, min_confidence=0.0, min_interval_s=1.0, on_message=print):
    meta, scaler, model = load_artifacts("fault")
    next_meta, next_scaler, next_model = load_artifacts("fault_next")
    horizon = next_meta["horizon"]

    buffer = LiveWindowBuffer(meta)
    start_ts = None
    last_scored_ts = None
    n_rows = 0

    for row in source:
        buffer.push(row)
        n_rows += 1
        if start_ts is None:
            start_ts = row["timestamp"]

        window, end_ts = buffer.latest_window()
        if window is None:
            continue
        if last_scored_ts is not None and (end_ts - last_scored_ts) / 1e6 < min_interval_s:
            continue
        last_scored_ts = end_ts

        current_label, current_conf = score_window(window, scaler, model, meta)
        next_label, next_conf = score_window(window, next_scaler, next_model, next_meta)
        if current_conf < min_confidence or next_conf < min_confidence:
            continue

        elapsed = (end_ts - start_ts) / 1e6
        avg_dt = elapsed / max(n_rows - 1, 1)
        horizon_seconds = horizon * avg_dt
        msg = advisory(current_label, current_conf, next_label, next_conf, horizon_seconds)

        line = (
            f"t={elapsed:6.1f}s  fault={current_label:<14} ({current_conf:.2f})  "
            f"next~{horizon_seconds:4.1f}s={next_label:<14} ({next_conf:.2f})"
        )
        if msg:
            line += f"   {msg}"
        on_message(line)


def main():
    parser = argparse.ArgumentParser(
        description="Live decision-support inference (advisory only - does not control the vehicle)."
    )
    parser.add_argument("--mode", choices=["replay", "mavlink"], default="replay")
    parser.add_argument("--log", default="data/sitl_motor_out_1.bin", help="Log to replay (--mode replay)")
    parser.add_argument("--speed", type=float, default=20.0, help="Replay speed multiplier, 0=no sleep (--mode replay)")
    parser.add_argument("--connection", default="udp:127.0.0.1:14550", help="MAVLink connection string (--mode mavlink)")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--min-interval", type=float, default=1.0, help="Seconds of flight-time between scored windows")
    parser.add_argument("--max-rows", type=int, default=None, help="Stop after this many rows (demos/tests)")
    args = parser.parse_args()

    source = (
        replay_log_source(args.log, speed=args.speed)
        if args.mode == "replay"
        else mavlink_source(args.connection)
    )
    if args.max_rows is not None:
        from itertools import islice
        source = islice(source, args.max_rows)
    run(source, min_confidence=args.min_confidence, min_interval_s=args.min_interval)


if __name__ == "__main__":
    main()
