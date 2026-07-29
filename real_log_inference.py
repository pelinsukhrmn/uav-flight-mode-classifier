import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import pandas as pd

from flight_mode_inference import (
    load_flight_log, load_artifacts, predict, summarize_segments, build_timeline_figure,
    load_ground_truth, attach_ground_truth, evaluate_predictions,
)

LOG_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/sample.ulg"

print(f"Loading real PX4 flight log: {LOG_PATH}")
data = load_flight_log(LOG_PATH)

ground_truth_df = load_ground_truth(LOG_PATH)
if ground_truth_df is not None:
    data = attach_ground_truth(data, ground_truth_df)

duration_seconds = (data["timestamp"].iloc[-1] - data["timestamp"].iloc[0]) / 1e6
print(f"Flight duration: {duration_seconds:.1f}s, samples: {len(data)}")

meta, scaler, model = load_artifacts()
result = predict(data, meta, scaler, model)

if result is None:
    print("Not enough samples in this log to build a single window.")
    sys.exit(0)

predicted_labels = result["predicted_labels"]
confidences = result["confidences"]

print("\nPredicted flight mode distribution on real telemetry:")
print(pd.Series(predicted_labels).value_counts())
print(f"\nMean prediction confidence: {confidences.mean():.3f}")

evaluation = evaluate_predictions(data, result) if ground_truth_df is not None else None
if evaluation is not None:
    print(f"\nGround-truth accuracy (vs PX4's own nav_state): {evaluation['accuracy'] * 100:.1f}% "
          f"({evaluation['coverage'] * 100:.1f}% of flight covered by an unambiguous nav_state, "
          f"{evaluation['n_evaluated']} windows)")
else:
    print("\nNo PX4 nav_state ground truth available for this log (no vehicle_status topic, "
          "or none of its nav_states map to our labels).")

segments = summarize_segments(result["window_times"], predicted_labels, confidences)

print("\nPredicted segments:")
print(f"  {'mode':<10} {'start_s':>10} {'end_s':>10} {'duration_s':>12} {'confidence':>12}")
for seg in segments:
    print(f"  {seg['mode']:<10} {seg['start_s']:>10.2f} {seg['end_s']:>10.2f} "
          f"{seg['duration_s']:>12.2f} {seg['mean_confidence']:>12.3f}")

fig = build_timeline_figure(data, result, LOG_PATH)
output_path = f"{Path(LOG_PATH).stem}_flight_mode_timeline.png"
fig.savefig(output_path, dpi=150)
print(f"\nSaved timeline plot to {output_path}")
