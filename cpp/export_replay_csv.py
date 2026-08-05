# Bir ArduPilot .bin logunun ham özellik satırlarını main_replay.cpp'nin okuyacağı bir CSV'ye döker.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import os
os.chdir(Path(__file__).resolve().parent.parent)

from ardupilot_log import load_flight_log

log_path = sys.argv[1] if len(sys.argv) > 1 else "data/sitl_motor_out_1.bin"
data = load_flight_log(log_path)

out_path = Path(__file__).resolve().parent / "replay_data.csv"
with open(out_path, "w") as f:
    f.write("timestamp,vertical_speed,horizontal_speed,roll_angle,pitch_angle\n")
    for _, row in data.iterrows():
        f.write(f"{row['timestamp']},{row['vertical_speed']},{row['horizontal_speed']},"
                f"{row['roll_angle']},{row['pitch_angle']}\n")

print(f"Wrote {out_path}: {len(data)} rows from {log_path}")
