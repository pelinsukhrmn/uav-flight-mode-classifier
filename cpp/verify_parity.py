"""Dump raw windows + the real Keras models' reference predictions, so the
hand-rolled C++ forward pass (lstm_model.hpp) can be checked for numerical
parity against the actual trained models on real flight data.

Run from the repo root: python cpp/verify_parity.py
Writes cpp/parity_data.txt, then run the compiled parity_check to compare.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os
os.chdir(Path(__file__).resolve().parent.parent)

from flight_mode_inference import load_flight_log, load_artifacts, build_windows

N_WINDOWS = 100
LOG = "data/real_flight_2.ulg"

data = load_flight_log(LOG)

lines = []
for prefix in ["flight_mode", "flight_mode_next"]:
    meta, scaler, model = load_artifacts(prefix)
    windows, window_end_idx = build_windows(data.copy(), meta)
    windows = windows[:N_WINDOWS]

    n, w, f = windows.shape
    scaled = scaler.transform(windows.reshape(-1, f)).reshape(n, w, f)
    probs = model.predict(scaled, verbose=0)

    lines.append(f"MODEL {prefix} {n} {f} {w} {len(meta['classes'])}")
    for i in range(n):
        row_floats = " ".join(f"{v:.9g}" for v in windows[i].flatten())
        prob_floats = " ".join(f"{v:.9g}" for v in probs[i])
        lines.append(f"{row_floats} | {prob_floats}")

out_path = Path(__file__).resolve().parent / "parity_data.txt"
out_path.write_text("\n".join(lines) + "\n")
print(f"Wrote {out_path}: {N_WINDOWS} windows x 2 models from {LOG}")
