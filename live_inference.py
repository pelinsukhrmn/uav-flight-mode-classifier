"""Streaming decision-support inference.

Feeds raw feature rows one at a time (from a replayed .ulg log, or a live
MAVLink connection) into a rolling window buffer, and reports the current-mode
and next-mode forecasts as each new window completes.

This is advisory only: it prints/returns messages, it never sends anything
back to the vehicle. See README's "Future work" section for why direct
control is out of scope for this model.
"""
import argparse
import time
from collections import deque

import numpy as np
import pandas as pd

from flight_mode_inference import load_flight_log, load_artifacts, build_windows


def replay_log_source(log_path, speed=20.0):
    """Yield raw feature rows from a .ulg log, sleeping between them to mimic a live stream.

    speed=0 disables sleeping (fastest possible replay - used by tests/demos).
    speed=1.0 replays at the log's real sample rate; higher values are faster.
    """
    data = load_flight_log(log_path)
    prev_ts = None
    for _, row in data.iterrows():
        if speed and prev_ts is not None:
            dt = (row["timestamp"] - prev_ts) / 1e6
            if dt > 0:
                time.sleep(dt / speed)
        prev_ts = row["timestamp"]
        yield {
            "timestamp": row["timestamp"],
            "vertical_speed": row["vertical_speed"],
            "horizontal_speed": row["horizontal_speed"],
            "roll_angle": row["roll_angle"],
            "pitch_angle": row["pitch_angle"],
        }


def mavlink_source(connection_string="udp:127.0.0.1:14540"):
    """Yield raw feature rows from a live MAVLink connection (SITL or real telemetry radio).

    Mirrors the feature definitions in flight_mode_inference.load_flight_log, but
    reads them from ATTITUDE (roll/pitch already in radians, no quaternion math
    needed) and LOCAL_POSITION_NED (vx/vy/vz) instead of a .ulg file.

    NOT exercised in this environment - no pymavlink/SITL toolchain available here.
    Written against the documented MAVLink message fields; verify against a real
    SITL/vehicle connection before relying on it.
    """
    from pymavlink import mavutil

    conn = mavutil.mavlink_connection(connection_string)
    conn.wait_heartbeat()

    latest_attitude = None
    while True:
        msg = conn.recv_match(type=["ATTITUDE", "LOCAL_POSITION_NED"], blocking=True)
        if msg is None:
            continue
        if msg.get_type() == "ATTITUDE":
            latest_attitude = msg
            continue
        if latest_attitude is None:
            continue
        yield {
            "timestamp": msg.time_boot_ms * 1000,
            "vertical_speed": -msg.vz,
            "horizontal_speed": (msg.vx ** 2 + msg.vy ** 2) ** 0.5,
            "roll_angle": np.degrees(latest_attitude.roll),
            "pitch_angle": np.degrees(latest_attitude.pitch),
        }


class LiveWindowBuffer:
    """Rolling raw-feature history that produces the latest complete window on demand.

    Keeps window_size + 1 rows: the extra row gives the window's first sample a
    real predecessor to diff against, so its delta features match what an
    offline build_windows() call would produce for the same stretch of flight.
    """

    def __init__(self, meta):
        self.meta = meta
        self.rows = deque(maxlen=meta["window_size"] + 1)

    def push(self, row):
        self.rows.append(row)

    def latest_window(self):
        """Return (window[1, w, f], window_end_timestamp), or (None, None) if not full yet."""
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
    """A short advisory string when the forecast disagrees with the current mode, else None."""
    if next_label == current_label:
        return None
    urgent = next_label in ("anomaly", "land", "rtl") and next_conf >= 0.6
    level = "UYARI" if urgent else "bilgi"
    return (
        f"[{level}] su an: {current_label} ({current_conf:.2f}) -> "
        f"~{horizon_seconds:.1f}s sonra tahmin: {next_label} ({next_conf:.2f})"
    )


def run(source, min_confidence=0.0, min_interval_s=1.0, on_message=print):
    """min_interval_s throttles how often windows actually get scored (by flight time,
    not wall-clock) - MAVLink attitude/position messages can arrive tens of times a
    second, and mode changes don't need re-scoring on every single one of them. This
    also keeps Keras's per-call overhead (each .predict() call costs tens of ms
    regardless of batch size) from dominating runtime on a fast replay."""
    meta, scaler, model = load_artifacts("flight_mode")
    next_meta, next_scaler, next_model = load_artifacts("flight_mode_next")
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
            f"t={elapsed:6.1f}s  mode={current_label:<10} ({current_conf:.2f})  "
            f"next~{horizon_seconds:4.1f}s={next_label:<10} ({next_conf:.2f})"
        )
        if msg:
            line += f"   {msg}"
        on_message(line)


def main():
    parser = argparse.ArgumentParser(
        description="Live decision-support inference (advisory only - does not control the vehicle)."
    )
    parser.add_argument("--mode", choices=["replay", "mavlink"], default="replay")
    parser.add_argument("--log", default="data/real_flight_2.ulg", help="Log to replay (--mode replay)")
    parser.add_argument("--speed", type=float, default=20.0, help="Replay speed multiplier, 0=no sleep (--mode replay)")
    parser.add_argument("--connection", default="udp:127.0.0.1:14540", help="MAVLink connection string (--mode mavlink)")
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
