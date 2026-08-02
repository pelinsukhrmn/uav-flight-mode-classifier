"""Dump a .ulg log's raw feature rows to a CSV for main_replay.cpp to stream -
the C++-side equivalent of live_inference.py's replay_log_source, since the
C++ prototype has no pyulog equivalent to parse .ulg files directly.

Usage: python cpp/export_replay_csv.py data/sample.ulg
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import os
os.chdir(Path(__file__).resolve().parent.parent)

from flight_mode_inference import load_flight_log

log_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample.ulg"
data = load_flight_log(log_path)

out_path = Path(__file__).resolve().parent / "replay_data.csv"
with open(out_path, "w") as f:
    f.write("timestamp,vertical_speed,horizontal_speed,roll_angle,pitch_angle\n")
    for _, row in data.iterrows():
        f.write(f"{row['timestamp']},{row['vertical_speed']},{row['horizontal_speed']},"
                f"{row['roll_angle']},{row['pitch_angle']}\n")

print(f"Wrote {out_path}: {len(data)} rows from {log_path}")
