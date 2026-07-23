import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

np.random.seed(42)
n_per_mode = 300
n_transition = 300

mode_params = {
    "hover": {
        "vertical_speed": (0.0, 0.15),
        "horizontal_speed": (0.3, 0.2),
        "roll_angle": (0.0, 1.0),
        "pitch_angle": (0.0, 1.0),
    },
    "takeoff": {
        "vertical_speed": (2.0, 0.5),
        "horizontal_speed": (0.2, 0.15),
        "roll_angle": (0.0, 1.0),
        "pitch_angle": (2.0, 1.5),
    },
    "ascend": {
        "vertical_speed": (3.0, 0.8),
        "horizontal_speed": (1.0, 0.6),
        "roll_angle": (0.0, 2.0),
        "pitch_angle": (8.0, 2.5),
    },
    "cruise": {
        "vertical_speed": (0.0, 0.3),
        "horizontal_speed": (10.0, 2.5),
        "roll_angle": (5.0, 4.0),
        "pitch_angle": (5.0, 3.0),
    },
    "rtl": {
        "vertical_speed": (-0.5, 0.4),
        "horizontal_speed": (12.0, 2.5),
        "roll_angle": (2.0, 2.5),
        "pitch_angle": (6.0, 3.0),
    },
    "descend": {
        "vertical_speed": (-3.0, 0.8),
        "horizontal_speed": (1.0, 0.6),
        "roll_angle": (0.0, 2.0),
        "pitch_angle": (-8.0, 2.5),
    },
    "land": {
        "vertical_speed": (-1.2, 0.4),
        "horizontal_speed": (0.2, 0.15),
        "roll_angle": (0.0, 1.0),
        "pitch_angle": (-2.0, 1.5),
    },
}

MODE_CYCLE = ["hover", "takeoff", "ascend", "cruise", "rtl", "descend", "land"]
transition_pairs = [(MODE_CYCLE[i], MODE_CYCLE[(i + 1) % len(MODE_CYCLE)]) for i in range(len(MODE_CYCLE))]

def generate_mode(mode, params, n):
    return pd.DataFrame({
        "vertical_speed": np.random.normal(params["vertical_speed"][0], params["vertical_speed"][1], n),
        "horizontal_speed": np.random.normal(params["horizontal_speed"][0], params["horizontal_speed"][1], n),
        "roll_angle": np.random.normal(params["roll_angle"][0], params["roll_angle"][1], n),
        "pitch_angle": np.random.normal(params["pitch_angle"][0], params["pitch_angle"][1], n),
        "mode": mode,
    })

def generate_transitions(pairs, params, n):
    features = ["vertical_speed", "horizontal_speed", "roll_angle", "pitch_angle"]
    pair_indices = np.random.randint(0, len(pairs), n)
    alphas = np.random.uniform(0.3, 0.7, n)
    rows = []
    for pair_index, alpha in zip(pair_indices, alphas):
        mode_a, mode_b = pairs[pair_index]
        row = {}
        for feature in features:
            mean_a, std_a = params[mode_a][feature]
            mean_b, std_b = params[mode_b][feature]
            blended_mean = alpha * mean_a + (1 - alpha) * mean_b
            noise_std = max(std_a, std_b) * 1.5
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
