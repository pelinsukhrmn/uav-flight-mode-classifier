# Gercek SITL loglarinda sinif bazli precision/recall ve kontrol ucuslarindaki yanlis alarm oranini olcer.
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ardupilot_log import load_flight_log, load_fault_ground_truth
from inference_common import attach_ground_truth, load_artifacts, predict

SUSTAIN_WINDOWS = 3


def sustained_alarms(labels, fault):
    run = 0
    alarms = 0
    for label in labels:
        run = run + 1 if label == fault else 0
        if run == SUSTAIN_WINDOWS:
            alarms += 1
    return alarms


def main():
    parser = argparse.ArgumentParser(description="Per-class evaluation of the production model on real SITL logs.")
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--prefix", default="fault")
    args = parser.parse_args()

    meta, scaler, model = load_artifacts(args.prefix)
    all_true, all_pred = [], []
    control_counts = Counter()
    control_alarms = Counter()
    control_windows = 0

    for log_path in sorted(args.logs):
        data = load_flight_log(log_path)
        result = predict(data, meta, scaler, model)
        if result is None:
            continue
        labels = result["predicted_labels"]

        ground_truth_df = load_fault_ground_truth(log_path)
        if ground_truth_df is None:
            continue
        labelled = attach_ground_truth(data, ground_truth_df)
        truth = labelled["ground_truth_mode"].to_numpy()[-len(labels):]

        keep = ~np.equal(truth, None)
        all_true.extend(truth[keep].tolist())
        all_pred.extend(np.asarray(labels)[keep].tolist())

        if set(truth[keep]) == {"normal"}:
            control_windows += len(labels)
            control_counts.update(labels)
            for fault in meta["classes"]:
                if fault != "normal":
                    control_alarms[fault] += sustained_alarms(labels, fault)

    print(classification_report(all_true, all_pred, digits=3, zero_division=0))

    if control_windows:
        print(f"Fault-free control flights ({control_windows} windows):")
        for label, count in control_counts.most_common():
            print(f"  predicted {label:18s} {count:6d}  ({100 * count / control_windows:5.1f}%)")
        print(f"\nSustained ({SUSTAIN_WINDOWS} consecutive windows) false alarms on fault-free flights:")
        for fault, count in sorted(control_alarms.items(), key=lambda item: -item[1]):
            print(f"  {fault:18s} {count}")


if __name__ == "__main__":
    main()
