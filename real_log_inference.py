import sys
import numpy as np
import pandas as pd
import joblib
from pyulog import ULog
from tensorflow import keras

LOG_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/sample.ulg"

def quaternion_to_roll_pitch(q0, q1, q2, q3):
    sinr_cosp = 2 * (q0 * q1 + q2 * q3)
    cosr_cosp = 1 - 2 * (q1 * q1 + q2 * q2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = np.clip(2 * (q0 * q2 - q3 * q1), -1.0, 1.0)
    pitch = np.arcsin(sinp)

    return np.degrees(roll), np.degrees(pitch)

print(f"Loading real PX4 flight log: {LOG_PATH}")
log = ULog(LOG_PATH)

local_position = log.get_dataset("vehicle_local_position")
attitude = log.get_dataset("vehicle_attitude")

position_df = pd.DataFrame({
    "timestamp": local_position.data["timestamp"],
    "vertical_speed": -local_position.data["vz"],
    "horizontal_speed": np.sqrt(local_position.data["vx"] ** 2 + local_position.data["vy"] ** 2),
}).sort_values("timestamp")

roll, pitch = quaternion_to_roll_pitch(
    attitude.data["q[0]"], attitude.data["q[1]"], attitude.data["q[2]"], attitude.data["q[3]"]
)
attitude_df = pd.DataFrame({
    "timestamp": attitude.data["timestamp"],
    "roll_angle": roll,
    "pitch_angle": pitch,
}).sort_values("timestamp")

data = pd.merge_asof(position_df, attitude_df, on="timestamp", direction="nearest")

duration_seconds = (data["timestamp"].iloc[-1] - data["timestamp"].iloc[0]) / 1e6
print(f"Flight duration: {duration_seconds:.1f}s, samples: {len(data)}")

meta = joblib.load("flight_mode_meta.joblib")
scaler = joblib.load("flight_mode_scaler.joblib")
model = keras.models.load_model("flight_mode_model.keras")

classes = meta["classes"]
features = meta["features"]
window_size = meta["window_size"]

base_features = [f for f in features if not f.endswith("_delta")]
for feature in base_features:
    delta_feature = f"{feature}_delta"
    if delta_feature in features:
        data[delta_feature] = data[feature].diff().fillna(0)

X = data[features].to_numpy()

n_windows = len(X) - window_size + 1
if n_windows < 1:
    print("Not enough samples in this log to build a single window.")
    sys.exit(0)

windows = np.array([X[start:start + window_size] for start in range(n_windows)])
n, w, f = windows.shape
windows_scaled = scaler.transform(windows.reshape(-1, f)).reshape(n, w, f)

predictions = np.argmax(model.predict(windows_scaled, verbose=0), axis=1)
predicted_labels = [classes[i] for i in predictions]

print("\nPredicted flight mode distribution on real telemetry:")
print(pd.Series(predicted_labels).value_counts())

print("\nFirst 10 predictions:")
for i in range(min(10, len(predicted_labels))):
    print(f"  window {i}: {predicted_labels[i]}")
