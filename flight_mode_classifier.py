import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from flight_data import MODE_PARAMS as mode_params, MODE_CYCLE, TRAVEL_MODES, SPEED_FEATURES, SPEED_SCALE_RANGE, PITCH_TRIM_RANGE

np.random.seed(42)
n_per_mode = 300
n_transition = 300

transition_pairs = [(MODE_CYCLE[i], MODE_CYCLE[(i + 1) % len(MODE_CYCLE)]) for i in range(len(MODE_CYCLE))]

def generate_mode(mode, params, n):
    speed_scale = np.random.uniform(*SPEED_SCALE_RANGE, n) if mode in TRAVEL_MODES else np.ones(n)
    pitch_trim = np.random.uniform(*PITCH_TRIM_RANGE, n)
    return pd.DataFrame({
        "vertical_speed": np.random.normal(params["vertical_speed"][0], params["vertical_speed"][1], n) * speed_scale,
        "horizontal_speed": np.random.normal(params["horizontal_speed"][0], params["horizontal_speed"][1], n) * speed_scale,
        "roll_angle": np.random.normal(params["roll_angle"][0], params["roll_angle"][1], n),
        "pitch_angle": np.random.normal(params["pitch_angle"][0], params["pitch_angle"][1], n) + pitch_trim,
        "mode": mode,
    })

def generate_transitions(pairs, params, n):
    features = ["vertical_speed", "horizontal_speed", "roll_angle", "pitch_angle"]
    pair_indices = np.random.randint(0, len(pairs), n)
    alphas = np.random.uniform(0.3, 0.7, n)
    speed_scales = np.random.uniform(*SPEED_SCALE_RANGE, n)
    pitch_trims = np.random.uniform(*PITCH_TRIM_RANGE, n)
    rows = []
    for pair_index, alpha, speed_scale, pitch_trim in zip(pair_indices, alphas, speed_scales, pitch_trims):
        mode_a, mode_b = pairs[pair_index]
        involves_travel_mode = mode_a in TRAVEL_MODES or mode_b in TRAVEL_MODES
        row = {}
        for feature in features:
            mean_a, std_a = params[mode_a][feature]
            mean_b, std_b = params[mode_b][feature]
            blended_mean = alpha * mean_a + (1 - alpha) * mean_b
            noise_std = max(std_a, std_b) * 1.5
            if involves_travel_mode and feature in SPEED_FEATURES:
                blended_mean *= speed_scale
                noise_std *= speed_scale
            if feature == "pitch_angle":
                blended_mean += pitch_trim
            row[feature] = np.random.normal(blended_mean, noise_std)
        row["mode"] = "transition"
        rows.append(row)
    return pd.DataFrame(rows)

frames = [generate_mode(mode, params, n_per_mode) for mode, params in mode_params.items()]
frames.append(generate_transitions(transition_pairs, mode_params, n_transition))

data = pd.concat(frames, ignore_index=True)
data["horizontal_speed"] = data["horizontal_speed"].clip(lower=0)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print("Dataset shape:", data.shape)
print(data["mode"].value_counts())

X = data.drop("mode", axis=1)
y = data["mode"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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

sample_readings = pd.DataFrame([
    {"vertical_speed": 0.05, "horizontal_speed": 0.4, "roll_angle": 0.5, "pitch_angle": 0.3},
    {"vertical_speed": 0.1, "horizontal_speed": 10.5, "roll_angle": 5.0, "pitch_angle": 5.5},
    {"vertical_speed": 1.4, "horizontal_speed": 0.7, "roll_angle": 1.0, "pitch_angle": 4.5},
])
sample_predictions = model.predict(sample_readings)

print("\nSample Predictions:")
for i, prediction in enumerate(sample_predictions):
    print(f"Reading {i + 1}: {sample_readings.iloc[i].to_dict()}")
    print(f"Predicted mode: {prediction}\n")
