# Uretilen SITL loglarini ve etiket sidecar'larini egitime girmeden once denetler.
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from pymavlink import mavutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ardupilot_log import load_flight_log, _sidecar_path

MIN_DURATION_S = 60.0
MAX_DURATION_S = 400.0
MIN_AIRBORNE_ALT_M = 1.0
FEATURE_LIMITS = {
    "roll_angle": (-180.0, 180.0),
    "pitch_angle": (-180.0, 180.0),
    "vertical_speed": (-30.0, 30.0),
    "horizontal_speed": (0.0, 60.0),
}


def load_altitude(log_path):
    log = mavutil.mavlink_connection(str(log_path))
    rows = []
    while True:
        msg = log.recv_match(type=["CTUN"], blocking=False)
        if msg is None:
            break
        rows.append({"timestamp": msg.TimeUS, "alt": msg.Alt})
    return pd.DataFrame(rows)


def check_log(log_path):
    problems = []
    sidecar = _sidecar_path(log_path)
    if not sidecar.exists():
        return [f"no sidecar at {sidecar.name}"], None

    windows = json.loads(sidecar.read_text())
    data = load_flight_log(log_path)
    if data.empty:
        return ["log has no ATT/CTUN/GPS rows"], None

    elapsed = (data["timestamp"] - data["timestamp"].iloc[0]) / 1e6
    duration = float(elapsed.iloc[-1])

    if not MIN_DURATION_S <= duration <= MAX_DURATION_S:
        problems.append(f"duration {duration:.0f}s outside [{MIN_DURATION_S:.0f},{MAX_DURATION_S:.0f}]s - likely several flights in one log")

    for name, (low, high) in FEATURE_LIMITS.items():
        col = data[name]
        if col.min() < low or col.max() > high:
            problems.append(f"{name} range [{col.min():.1f},{col.max():.1f}] outside [{low},{high}] - unit error?")

    altitude = load_altitude(log_path)
    for window in windows:
        if window["start_s"] < 0 or window["end_s"] > duration:
            problems.append(f"fault window {window['start_s']:.0f}-{window['end_s']:.0f}s falls outside the {duration:.0f}s log")
            continue
        if not altitude.empty:
            alt_elapsed = (altitude["timestamp"] - data["timestamp"].iloc[0]) / 1e6
            in_window = (alt_elapsed >= window["start_s"]) & (alt_elapsed < window["end_s"])
            if in_window.any() and altitude.loc[in_window, "alt"].max() < MIN_AIRBORNE_ALT_M:
                problems.append(f"vehicle never above {MIN_AIRBORNE_ALT_M}m during the {window['fault']} window - fault injected on the ground")

    signature = (float(data["timestamp"].iloc[0]), round(duration, 1))
    return problems, signature


def main():
    parser = argparse.ArgumentParser(description="Validate generated SITL fault logs before training on them.")
    parser.add_argument("logs", nargs="+", help=".bin logs to check")
    args = parser.parse_args()

    signatures = {}
    failed = 0
    for log_path in sorted(args.logs):
        problems, signature = check_log(Path(log_path))
        if signature is not None:
            if signature in signatures:
                problems.append(f"identical start time and duration to {signatures[signature]} - same underlying log copied twice")
            else:
                signatures[signature] = Path(log_path).name
        name = Path(log_path).name
        if problems:
            failed += 1
            print(f"FAIL {name}")
            for problem in problems:
                print(f"       {problem}")
        else:
            print(f"ok   {name}")

    print(f"\n{len(args.logs) - failed}/{len(args.logs)} logs usable")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
