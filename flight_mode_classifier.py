import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

np.random.seed(42)
n_per_mode = 300

def generate_mode(mode, vertical_speed, horizontal_speed, roll_angle, pitch_angle, n):
    return pd.DataFrame({
        "vertical_speed": np.random.normal(vertical_speed[0], vertical_speed[1], n),
        "horizontal_speed": np.clip(np.random.normal(horizontal_speed[0], horizontal_speed[1], n), 0, None),
        "roll_angle": np.random.normal(roll_angle[0], roll_angle[1], n),
        "pitch_angle": np.random.normal(pitch_angle[0], pitch_angle[1], n),
        "mode": mode,
    })

hover = generate_mode("hover", (0.0, 0.15), (0.3, 0.2), (0.0, 1.0), (0.0, 1.0), n_per_mode)
ascend = generate_mode("ascend", (3.0, 0.8), (1.0, 0.6), (0.0, 2.0), (8.0, 2.5), n_per_mode)
descend = generate_mode("descend", (-3.0, 0.8), (1.0, 0.6), (0.0, 2.0), (-8.0, 2.5), n_per_mode)
cruise = generate_mode("cruise", (0.0, 0.3), (10.0, 2.5), (5.0, 4.0), (5.0, 3.0), n_per_mode)

data = pd.concat([hover, ascend, descend, cruise], ignore_index=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print("Dataset shape:", data.shape)
print(data["mode"].value_counts())

X = data.drop("mode", axis=1)
y = data["mode"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nTraining Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
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
])
sample_predictions = model.predict(sample_readings)

print("\nSample Predictions:")
for i, prediction in enumerate(sample_predictions):
    print(f"Reading {i + 1}: {sample_readings.iloc[i].to_dict()}")
    print(f"Predicted mode: {prediction}\n")
