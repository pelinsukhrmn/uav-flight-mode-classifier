# Bağımsız (pencere/hafızasız) tek okumadan arıza sınıflandıran Random Forest referans modeli.
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from flight_data import MODE_PARAMS as mode_params, MODE_CYCLE, TRAVEL_MODES, SPEED_SCALE_RANGE, PITCH_TRIM_RANGE, randomize_mode_params
from fault_injection import FAULT_CLASSES, FAULT_GENERATORS

rng = np.random.default_rng(42)
n_per_mode = 300
n_per_fault = 300
n_fault_realizations = 15


def generate_normal(mode, params, n, rng):
    speed_scale = rng.uniform(*SPEED_SCALE_RANGE, n) if mode in TRAVEL_MODES else np.ones(n)
    pitch_trim = rng.uniform(*PITCH_TRIM_RANGE, n)
    df = pd.DataFrame({
        "vertical_speed": rng.normal(params["vertical_speed"][0], params["vertical_speed"][1], n) * speed_scale,
        "horizontal_speed": rng.normal(params["horizontal_speed"][0], params["horizontal_speed"][1], n) * speed_scale,
        "roll_angle": rng.normal(params["roll_angle"][0], params["roll_angle"][1], n),
        "pitch_angle": rng.normal(params["pitch_angle"][0], params["pitch_angle"][1], n) + pitch_trim,
        "fault": "normal",
    })
    df["group_id"] = [f"normal_{mode}_{i}" for i in range(n)]
    return df


def generate_fault_realization(fault_type, n, rng):
    mode = rng.choice(MODE_CYCLE)
    speed_scale = rng.uniform(*SPEED_SCALE_RANGE)
    pitch_trim = rng.uniform(*PITCH_TRIM_RANGE)
    randomized = randomize_mode_params(mode_params, speed_scale, pitch_trim)
    values = FAULT_GENERATORS[fault_type](n, rng, randomized[mode])
    df = pd.DataFrame(values)
    df["fault"] = fault_type
    return df


def generate_fault(fault_type, total_n, rng, n_realizations=n_fault_realizations):
    per_realization = max(1, total_n // n_realizations)
    frames = []
    for realization_id in range(n_realizations):
        df = generate_fault_realization(fault_type, per_realization, rng)
        df["group_id"] = f"{fault_type}_{realization_id}"
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


frames = [generate_normal(mode, mode_params[mode], n_per_mode, rng) for mode in MODE_CYCLE]
frames += [generate_fault(fault_type, n_per_fault, rng) for fault_type in FAULT_CLASSES]

data = pd.concat(frames, ignore_index=True)
data["horizontal_speed"] = data["horizontal_speed"].clip(lower=0)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print("Dataset shape:", data.shape)
print(data["fault"].value_counts())

X = data.drop(["fault", "group_id"], axis=1)
y = data["fault"]

train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, groups=data["group_id"]))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print("\nTraining Random Forest model...")
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions, labels=model.classes_))
print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nFeature Importances:")
for feature, importance in zip(X.columns, model.feature_importances_):
    print(f"  {feature:<18} {importance:.4f}")
