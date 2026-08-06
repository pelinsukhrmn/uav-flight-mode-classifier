# Modelin arizayi ArduPilot'un kendi ERR tespitinden ne kadar once adlandirdigini olcer.
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from pymavlink import mavutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ardupilot_log import load_flight_log, _sidecar_path, ERR_SUBSYS_TO_FAULT
from inference_common import load_artifacts, predict

CONSECUTIVE_WINDOWS = 3


def autopilot_detection_s(log_path, log_t0_us, fault):
    log = mavutil.mavlink_connection(str(log_path))
    while True:
        msg = log.recv_match(type=["ERR"], blocking=False)
        if msg is None:
            return None
        if ERR_SUBSYS_TO_FAULT.get(msg.Subsys) == fault and msg.ECode != 0:
            return (msg.TimeUS - log_t0_us) / 1e6


def first_sustained_detection_s(labels, elapsed_s, fault, after_s):
    run = 0
    for label, t in zip(labels, elapsed_s):
        if t < after_s:
            continue
        run = run + 1 if label == fault else 0
        if run >= CONSECUTIVE_WINDOWS:
            return t
    return None


def main():
    parser = argparse.ArgumentParser(description="Measure model lead time over ArduPilot's own fault detection.")
    parser.add_argument("logs", nargs="+")
    args = parser.parse_args()

    meta, scaler, model = load_artifacts("fault")
    print(f"{'log':32s} {'fault':16s} {'inject':>8s} {'model':>8s} {'ardupilot':>10s} {'lead':>8s}")

    leads = []
    for log_path in sorted(args.logs):
        sidecar = _sidecar_path(Path(log_path))
        windows = json.loads(sidecar.read_text()) if sidecar.exists() else []
        if not windows:
            continue
        fault = windows[0]["fault"]
        inject_s = windows[0]["start_s"]

        data = load_flight_log(log_path)
        result = predict(data, meta, scaler, model)
        if result is None:
            continue
        labels = result["predicted_labels"]
        elapsed_s = np.asarray(result["window_times"])

        model_s = first_sustained_detection_s(labels, elapsed_s, fault, inject_s)
        autopilot_s = autopilot_detection_s(log_path, data["timestamp"].iloc[0], fault)

        lead = None
        if model_s is not None and autopilot_s is not None:
            lead = autopilot_s - model_s
            leads.append(lead)
        print(f"{Path(log_path).name:32s} {fault:16s} {inject_s:8.1f} "
              f"{'-' if model_s is None else f'{model_s:8.1f}'} "
              f"{'-' if autopilot_s is None else f'{autopilot_s:10.1f}'} "
              f"{'-' if lead is None else f'{lead:8.1f}'}")

    if leads:
        print(f"\nmean lead time over ArduPilot's own detection: {np.mean(leads):.1f}s ({len(leads)} flights)")


if __name__ == "__main__":
    main()
