# C++ ileri geçişinin gerçek modellerle sayısal olarak eşleştiğini doğrulamak için referans veri üretir.
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import os
os.chdir(Path(__file__).resolve().parent.parent)

from inference_common import load_artifacts, build_windows
from ardupilot_log import load_flight_log

N_WINDOWS = 100
LOG = "data/sitl_motor_out_1.bin"

data = load_flight_log(LOG)

lines = []
for prefix in ["fault", "fault_next"]:
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
