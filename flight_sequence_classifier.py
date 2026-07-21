import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

FEATURES = ["vertical_speed", "horizontal_speed", "roll_angle", "pitch_angle"]
CYCLE = ["hover", "ascend", "cruise", "descend"]
WINDOW_SIZE = 10
N_FLIGHTS = 120
N_TRAIN_FLIGHTS = 96

MODE_PARAMS = {
    "hover": {"vertical_speed": (0.0, 0.15), "horizontal_speed": (0.3, 0.2), "roll_angle": (0.0, 1.0), "pitch_angle": (0.0, 1.0)},
    "ascend": {"vertical_speed": (3.0, 0.8), "horizontal_speed": (1.0, 0.6), "roll_angle": (0.0, 2.0), "pitch_angle": (8.0, 2.5)},
    "cruise": {"vertical_speed": (0.0, 0.3), "horizontal_speed": (10.0, 2.5), "roll_angle": (5.0, 4.0), "pitch_angle": (5.0, 3.0)},
    "descend": {"vertical_speed": (-3.0, 0.8), "horizontal_speed": (1.0, 0.6), "roll_angle": (0.0, 2.0), "pitch_angle": (-8.0, 2.5)},
}

def sample_mode_segment(mode, n, rng):
    params = MODE_PARAMS[mode]
    return {f: rng.normal(params[f][0], params[f][1], n) for f in FEATURES}

def sample_transition_segment(mode_a, mode_b, n, rng):
    params_a, params_b = MODE_PARAMS[mode_a], MODE_PARAMS[mode_b]
    alphas = np.linspace(1.0, 0.0, n)
    out = {}
    for f in FEATURES:
        mean_a, std_a = params_a[f]
        mean_b, std_b = params_b[f]
        blended_mean = alphas * mean_a + (1 - alphas) * mean_b
        noise_std = max(std_a, std_b) * 1.2
        out[f] = rng.normal(blended_mean, noise_std)
    return out

def sample_anomaly_segment(n, rng):
    return {
        "vertical_speed": rng.normal(0, 5.0, n),
        "horizontal_speed": np.abs(rng.normal(8, 6.0, n)),
        "roll_angle": rng.normal(0, 15.0, n),
        "pitch_angle": rng.normal(0, 15.0, n),
    }

def generate_flight(flight_id, rng, anomaly_prob=0.35):
    segments = []
    for i, mode in enumerate(CYCLE):
        n = int(rng.integers(15, 30))
        segments.append((mode, sample_mode_segment(mode, n, rng)))
        next_mode = CYCLE[(i + 1) % len(CYCLE)]
        n_t = int(rng.integers(5, 10))
        segments.append(("transition", sample_transition_segment(mode, next_mode, n_t, rng)))

    if rng.random() < anomaly_prob:
        n_a = int(rng.integers(5, 15))
        insert_at = int(rng.integers(0, len(segments) + 1))
        segments.insert(insert_at, ("anomaly", sample_anomaly_segment(n_a, rng)))

    labels = []
    feature_arrays = {f: [] for f in FEATURES}
    for mode, values in segments:
        n = len(values[FEATURES[0]])
        labels.extend([mode] * n)
        for f in FEATURES:
            feature_arrays[f].append(values[f])

    df = pd.DataFrame({f: np.concatenate(feature_arrays[f]) for f in FEATURES})
    df["mode"] = labels
    df["flight_id"] = flight_id
    return df

def build_windows(df, window_size, feature_columns):
    X_windows = []
    y_windows = []
    for _, group in df.groupby("flight_id"):
        features = group[feature_columns].to_numpy()
        labels = group["mode"].to_numpy()
        for start in range(len(group) - window_size + 1):
            X_windows.append(features[start:start + window_size])
            y_windows.append(labels[start + window_size - 1])
    return np.array(X_windows), np.array(y_windows)

rng = np.random.default_rng(42)
flights = [generate_flight(i, rng) for i in range(N_FLIGHTS)]
data = pd.concat(flights, ignore_index=True)

DELTA_FEATURES = [f"{feature}_delta" for feature in FEATURES]
for feature, delta_feature in zip(FEATURES, DELTA_FEATURES):
    data[delta_feature] = data.groupby("flight_id")[feature].diff().fillna(0)

ALL_FEATURES = FEATURES + DELTA_FEATURES

print("Total timesteps:", len(data))
print(data["mode"].value_counts())

train_df = data[data["flight_id"] < N_TRAIN_FLIGHTS]
test_df = data[data["flight_id"] >= N_TRAIN_FLIGHTS]

X_train, y_train_labels = build_windows(train_df, WINDOW_SIZE, ALL_FEATURES)
X_test, y_test_labels = build_windows(test_df, WINDOW_SIZE, ALL_FEATURES)

classes = sorted(data["mode"].unique())
label_to_index = {label: i for i, label in enumerate(classes)}
index_to_label = {i: label for label, i in label_to_index.items()}

y_train = np.array([label_to_index[label] for label in y_train_labels])
y_test = np.array([label_to_index[label] for label in y_test_labels])

n_train, window_size, n_features = X_train.shape
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(n_train, window_size, n_features)
X_test_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape[0], window_size, n_features)

class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

print("\nClass weights:")
for i, weight in class_weight_dict.items():
    print(f"  {index_to_label[i]:<12} {weight:.3f}")

def build_model(recurrent_layer):
    return keras.Sequential([
        keras.layers.Input(shape=(WINDOW_SIZE, n_features)),
        recurrent_layer,
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(len(classes), activation="softmax"),
    ])

architectures = {
    "LSTM": keras.layers.LSTM(32),
    "GRU": keras.layers.GRU(32),
}

results = {}
trained_models = {}
y_test_named = [index_to_label[i] for i in y_test]

for name, recurrent_layer in architectures.items():
    print(f"\nTraining {name} model...")
    model = build_model(recurrent_layer)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train_scaled, y_train, validation_split=0.15, epochs=20, batch_size=32, class_weight=class_weight_dict, verbose=2)

    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    predictions = np.argmax(model.predict(X_test_scaled, verbose=0), axis=1)
    predictions_named = [index_to_label[i] for i in predictions]

    results[name] = {
        "accuracy": test_accuracy,
        "confusion_matrix": confusion_matrix(y_test_named, predictions_named, labels=classes),
        "report": classification_report(y_test_named, predictions_named, labels=classes),
    }
    trained_models[name] = model

print("\n=== Architecture Comparison ===")
for name, result in results.items():
    print(f"\n{name} Test Accuracy: {result['accuracy'] * 100:.2f}%")
    print("Confusion Matrix:")
    print(result["confusion_matrix"])
    print("Classification Report:")
    print(result["report"])

best_name = max(results, key=lambda name: results[name]["accuracy"])
best_model = trained_models[best_name]

print(f"\nSaving best model ({best_name}) for reuse on real flight logs...")
best_model.save("flight_mode_model.keras")
joblib.dump(scaler, "flight_mode_scaler.joblib")
joblib.dump({"classes": classes, "features": ALL_FEATURES, "window_size": WINDOW_SIZE}, "flight_mode_meta.joblib")
