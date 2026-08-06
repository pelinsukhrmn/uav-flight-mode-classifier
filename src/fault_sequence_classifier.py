# Arıza-öngörü sıra modelini (nowcast + forecast) sentetik veriyle eğitir, SITL loglarıyla ince ayar yapar.
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from flight_data import FEATURES, MODE_CYCLE as CYCLE, MODE_PARAMS, SPEED_SCALE_RANGE, PITCH_TRIM_RANGE, randomize_mode_params
from fault_injection import FAULT_CLASSES, FAULT_GENERATORS
from inference_common import extract_labeled_real_windows, select_holdout_flights
from ardupilot_log import load_flight_log, load_fault_ground_truth

WINDOW_SIZE = 15
N_FLIGHTS = 200
N_TRAIN_FLIGHTS = 160
N_FAULTS_RANGE = (0, 2)
PREDICTION_HORIZON = 10

SITL_FINETUNE_LOGS = sorted(str(path) for path in Path("data").glob("sitl_*.bin"))
REAL_HOLDOUT_FRACTION = 0.3
REGRESSION_TOLERANCE = 0.03
FINETUNE_LR = 1e-4


def sample_mode_segment(mode, n, rng, mode_params):
    params = mode_params[mode]
    return {f: rng.normal(params[f][0], params[f][1], n) for f in FEATURES}


def sample_transition_segment(mode_a, mode_b, n, rng, mode_params):
    params_a, params_b = mode_params[mode_a], mode_params[mode_b]
    alphas = np.linspace(1.0, 0.0, n)
    out = {}
    for f in FEATURES:
        mean_a, std_a = params_a[f]
        mean_b, std_b = params_b[f]
        blended_mean = alphas * mean_a + (1 - alphas) * mean_b
        noise_std = max(std_a, std_b) * 1.2
        out[f] = rng.normal(blended_mean, noise_std)
    return out


def generate_flight(flight_id, rng, n_faults_range=N_FAULTS_RANGE):
    mission_scale = rng.uniform(*SPEED_SCALE_RANGE)
    pitch_trim = rng.uniform(*PITCH_TRIM_RANGE)
    mode_params = randomize_mode_params(MODE_PARAMS, mission_scale, pitch_trim)

    segments = []
    context_modes = []
    for i, mode in enumerate(CYCLE):
        n = int(rng.integers(15, 30))
        segments.append(("normal", sample_mode_segment(mode, n, rng, mode_params)))
        context_modes.append(mode)
        next_mode = CYCLE[(i + 1) % len(CYCLE)]
        n_t = int(rng.integers(5, 10))
        segments.append(("normal", sample_transition_segment(mode, next_mode, n_t, rng, mode_params)))
        context_modes.append(mode)

    n_faults = int(rng.integers(n_faults_range[0], n_faults_range[1] + 1))
    for _ in range(n_faults):
        fault_type = rng.choice(FAULT_CLASSES)
        insert_at = int(rng.integers(0, len(segments) + 1))
        context_mode = context_modes[min(insert_at, len(context_modes) - 1)]
        n_f = int(rng.integers(30, 70))
        fault_values = FAULT_GENERATORS[fault_type](n_f, rng, mode_params[context_mode])
        segments.insert(insert_at, (fault_type, fault_values))
        context_modes.insert(insert_at, context_mode)

    labels = []
    feature_arrays = {f: [] for f in FEATURES}
    for label, values in segments:
        n = len(values[FEATURES[0]])
        labels.extend([label] * n)
        for f in FEATURES:
            feature_arrays[f].append(values[f])

    df = pd.DataFrame({f: np.concatenate(feature_arrays[f]) for f in FEATURES})
    df["fault"] = labels
    df["flight_id"] = flight_id
    return df


def build_windows(df, window_size, feature_columns, horizon=0):
    X_windows = []
    y_windows = []
    for _, group in df.groupby("flight_id"):
        features = group[feature_columns].to_numpy()
        labels = group["fault"].to_numpy()
        for start in range(len(group) - window_size + 1 - horizon):
            X_windows.append(features[start:start + window_size])
            y_windows.append(labels[start + window_size - 1 + horizon])
    return np.array(X_windows), np.array(y_windows)


def build_model(feature_layers, n_features, n_classes):
    return keras.Sequential([
        keras.layers.Input(shape=(WINDOW_SIZE, n_features)),
        *feature_layers,
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(n_classes, activation="softmax"),
    ])


LSTM_UNITS = 64
RECURRENT_DROPOUT = 0.1
architectures = {
    "LSTM": lambda: [keras.layers.LSTM(LSTM_UNITS, recurrent_dropout=RECURRENT_DROPOUT)],
    "GRU": lambda: [keras.layers.GRU(LSTM_UNITS, recurrent_dropout=RECURRENT_DROPOUT)],
    "Conv1D": lambda: [
        keras.layers.Conv1D(LSTM_UNITS, kernel_size=3, activation="relu", padding="causal"),
        keras.layers.Dropout(0.1),
        keras.layers.Conv1D(LSTM_UNITS, kernel_size=3, activation="relu", padding="causal"),
        keras.layers.GlobalAveragePooling1D(),
    ],
    "BiLSTM": lambda: [keras.layers.Bidirectional(keras.layers.LSTM(LSTM_UNITS, recurrent_dropout=RECURRENT_DROPOUT))],
}


def early_stopping():
    return keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)


def fine_tune_on_real_data(model, model_prefix, scaler, classes, label_to_index,
                            X_train_scaled, y_train, base_test_acc, X_test_scaled, y_test,
                            n_features, horizon=0):
    print(f"\n=== SITL fine-tuning ({model_prefix}, horizon={horizon}) ===")
    meta = {"classes": classes, "features": ALL_FEATURES, "window_size": WINDOW_SIZE}
    real_train_windows, real_train_labels, real_holdout_windows, real_holdout_labels = [], [], [], []

    usable_logs = []
    for log_path in SITL_FINETUNE_LOGS:
        if not Path(log_path).exists():
            print(f"  {log_path}: not found, skipping")
            continue
        X_real, y_real = extract_labeled_real_windows(
            log_path, meta, load_flight_log, load_fault_ground_truth, horizon=horizon,
        )
        if X_real is None:
            print(f"  {log_path}: no usable fault ground truth, skipping")
            continue
        usable_logs.append((log_path, X_real, y_real))

    holdout_logs = set(select_holdout_flights([(path, y) for path, _, y in usable_logs], REAL_HOLDOUT_FRACTION))
    for log_path, X_real, y_real in usable_logs:
        if log_path in holdout_logs:
            print(f"  {log_path}: {len(X_real)} windows held out for evaluation (whole flight)")
            real_holdout_windows.append(X_real)
            real_holdout_labels.append(y_real)
        else:
            print(f"  {log_path}: {len(X_real)} windows for fine-tuning")
            real_train_windows.append(X_real)
            real_train_labels.append(y_real)

    if not real_train_windows or not any(len(w) for w in real_holdout_windows):
        print("No usable SITL ground-truth windows found - keeping the synthetic-only model as production.")
        return model

    X_real_train = np.concatenate(real_train_windows)
    y_real_train = np.array([label_to_index[label] for label in np.concatenate(real_train_labels)])
    X_real_holdout = np.concatenate(real_holdout_windows)
    y_real_holdout = np.array([label_to_index[label] for label in np.concatenate(real_holdout_labels)])

    X_real_train_scaled = scaler.transform(X_real_train.reshape(-1, n_features)).reshape(len(X_real_train), WINDOW_SIZE, n_features)
    X_real_holdout_scaled = scaler.transform(X_real_holdout.reshape(-1, n_features)).reshape(len(X_real_holdout), WINDOW_SIZE, n_features)

    pre_finetune_probabilities = model.predict(X_real_holdout_scaled, verbose=0)
    pre_finetune_real_acc = model.evaluate(X_real_holdout_scaled, y_real_holdout, verbose=0)[1]
    print(f"\nSynthetic-only model on SITL held-out windows: {pre_finetune_real_acc * 100:.2f}% "
          f"({len(X_real_holdout)} windows, classes: {sorted(set(np.concatenate(real_train_labels)) | set(np.concatenate(real_holdout_labels)))})")

    replay_size = min(len(X_train_scaled), 3 * len(X_real_train))
    if replay_size >= len(X_train_scaled):
        X_replay, y_replay = X_train_scaled, y_train
    else:
        _, X_replay, _, y_replay = train_test_split(
            X_train_scaled, y_train, test_size=replay_size, stratify=y_train, random_state=42,
        )
    X_finetune = np.concatenate([X_replay, X_real_train_scaled])
    y_finetune = np.concatenate([y_replay, y_real_train])

    finetune_class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_finetune), y=y_finetune)
    finetune_class_weight_dict = {i: weight for i, weight in enumerate(finetune_class_weights)}

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=FINETUNE_LR), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        X_finetune, y_finetune, validation_data=(X_real_holdout_scaled, y_real_holdout), epochs=15, batch_size=32,
        class_weight=finetune_class_weight_dict, callbacks=[early_stopping()], verbose=2,
    )

    post_finetune_real_acc = model.evaluate(X_real_holdout_scaled, y_real_holdout, verbose=0)[1]
    post_finetune_synthetic_acc = model.evaluate(X_test_scaled, y_test, verbose=0)[1]
    synthetic_regression = base_test_acc - post_finetune_synthetic_acc

    print(f"\nFine-tuned model on SITL held-out windows: {post_finetune_real_acc * 100:.2f}% "
          f"(was {pre_finetune_real_acc * 100:.2f}% before fine-tuning)")
    print(f"Fine-tuned model on synthetic test set: {post_finetune_synthetic_acc * 100:.2f}% "
          f"(was {base_test_acc * 100:.2f}% before fine-tuning, {synthetic_regression * 100:+.2f} points)")

    majority_class = int(np.bincount(y_real_holdout).argmax())
    majority_real_acc = np.bincount(y_real_holdout).max() / len(y_real_holdout)
    majority_macro_f1 = f1_score(y_real_holdout, np.full_like(y_real_holdout, majority_class),
                                 average="macro", zero_division=0)
    pre_macro_f1 = f1_score(y_real_holdout, np.argmax(pre_finetune_probabilities, axis=1),
                            average="macro", zero_division=0)
    post_macro_f1 = f1_score(y_real_holdout, np.argmax(model.predict(X_real_holdout_scaled, verbose=0), axis=1),
                             average="macro", zero_division=0)
    print(f"Majority-class baseline on the same SITL held-out windows: {majority_real_acc * 100:.2f}% "
          f"accuracy, {majority_macro_f1:.3f} macro-F1")
    print(f"Macro-F1 on SITL held-out windows: {pre_macro_f1:.3f} synthetic-only -> {post_macro_f1:.3f} fine-tuned")

    if post_macro_f1 <= majority_macro_f1:
        print(f"\nFine-tuned model does not beat the majority-class baseline on macro-F1 "
              f"({post_macro_f1:.3f} <= {majority_macro_f1:.3f}) - keeping the synthetic-only model as production.")
    elif post_macro_f1 >= pre_macro_f1 and synthetic_regression <= REGRESSION_TOLERANCE:
        print("\nFine-tuned model beats the majority-class baseline on macro-F1 and didn't regress "
              "synthetic accuracy beyond tolerance - saving it as production.")
        model.save(f"models/{model_prefix}_model.keras")
    else:
        print("\nFine-tuned model did not clear the acceptance bar - keeping the synthetic-only model as production.")
    return model


rng = np.random.default_rng(42)
flights = [generate_flight(i, rng) for i in range(N_FLIGHTS)]
data = pd.concat(flights, ignore_index=True)

DELTA_FEATURES = [f"{feature}_delta" for feature in FEATURES]
for feature, delta_feature in zip(FEATURES, DELTA_FEATURES):
    data[delta_feature] = data.groupby("flight_id")[feature].diff().fillna(0)

ALL_FEATURES = FEATURES + DELTA_FEATURES

print("Total timesteps:", len(data))
print(data["fault"].value_counts())

train_df = data[data["flight_id"] < N_TRAIN_FLIGHTS]
test_df = data[data["flight_id"] >= N_TRAIN_FLIGHTS]

X_train, y_train_labels = build_windows(train_df, WINDOW_SIZE, ALL_FEATURES)
X_test, y_test_labels = build_windows(test_df, WINDOW_SIZE, ALL_FEATURES)

classes = sorted(data["fault"].unique())
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
    print(f"  {index_to_label[i]:<16} {weight:.3f}")

print("\n=== Cross-Validation (flight-level, 5-fold) ===")
N_CV_FOLDS = 5
CV_EPOCHS = 30
flight_ids = data["flight_id"].unique()
kfold = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=42)

cv_scores = {name: [] for name in architectures}
for fold_idx, (train_pos, val_pos) in enumerate(kfold.split(flight_ids), start=1):
    fold_train_ids, fold_val_ids = flight_ids[train_pos], flight_ids[val_pos]
    fold_train_df = data[data["flight_id"].isin(fold_train_ids)]
    fold_val_df = data[data["flight_id"].isin(fold_val_ids)]

    X_fold_train, y_fold_train_labels = build_windows(fold_train_df, WINDOW_SIZE, ALL_FEATURES)
    X_fold_val, y_fold_val_labels = build_windows(fold_val_df, WINDOW_SIZE, ALL_FEATURES)
    y_fold_train = np.array([label_to_index[label] for label in y_fold_train_labels])
    y_fold_val = np.array([label_to_index[label] for label in y_fold_val_labels])

    fold_scaler = StandardScaler()
    n_fold_train = X_fold_train.shape[0]
    X_fold_train_scaled = fold_scaler.fit_transform(X_fold_train.reshape(-1, n_features)).reshape(n_fold_train, WINDOW_SIZE, n_features)
    X_fold_val_scaled = fold_scaler.transform(X_fold_val.reshape(-1, n_features)).reshape(X_fold_val.shape[0], WINDOW_SIZE, n_features)

    fold_class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_fold_train), y=y_fold_train)
    fold_class_weight_dict = {i: weight for i, weight in enumerate(fold_class_weights)}

    for name, make_layers in architectures.items():
        fold_model = build_model(make_layers(), n_features, len(classes))
        fold_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        fold_model.fit(
            X_fold_train_scaled, y_fold_train, epochs=CV_EPOCHS, batch_size=32,
            class_weight=fold_class_weight_dict, validation_data=(X_fold_val_scaled, y_fold_val),
            callbacks=[early_stopping()], verbose=0,
        )
        _, fold_accuracy = fold_model.evaluate(X_fold_val_scaled, y_fold_val, verbose=0)
        cv_scores[name].append(fold_accuracy)
        print(f"  Fold {fold_idx} {name}: {fold_accuracy * 100:.2f}%")

print("\nCross-validation summary (mean ± std across folds):")
for name, scores in cv_scores.items():
    scores = np.array(scores)
    print(f"  {name:<8} {scores.mean() * 100:.2f}% ± {scores.std() * 100:.2f}%")

results = {}
trained_models = {}
y_test_named = [index_to_label[i] for i in y_test]

for name, make_layers in architectures.items():
    print(f"\nTraining {name} model...")
    model = build_model(make_layers(), n_features, len(classes))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        X_train_scaled, y_train, validation_split=0.15, epochs=40, batch_size=32,
        class_weight=class_weight_dict, callbacks=[early_stopping()], verbose=2,
    )

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

best_name = "LSTM"
best_model = trained_models[best_name]

print(f"\nSaving best model ({best_name}) for reuse on real flight logs...")
best_model.save("models/fault_model.keras")
joblib.dump(scaler, "models/fault_scaler.joblib")
joblib.dump({"classes": classes, "features": ALL_FEATURES, "window_size": WINDOW_SIZE}, "models/fault_meta.joblib")

print(f"\n=== Fault forecaster ({best_name} architecture, {PREDICTION_HORIZON} steps ahead) ===")
X_train_next, y_train_next_labels = build_windows(train_df, WINDOW_SIZE, ALL_FEATURES, horizon=PREDICTION_HORIZON)
X_test_next, y_test_next_labels = build_windows(test_df, WINDOW_SIZE, ALL_FEATURES, horizon=PREDICTION_HORIZON)

y_train_next = np.array([label_to_index[label] for label in y_train_next_labels])
y_test_next = np.array([label_to_index[label] for label in y_test_next_labels])

n_train_next = X_train_next.shape[0]
next_scaler = StandardScaler()
X_train_next_scaled = next_scaler.fit_transform(X_train_next.reshape(-1, n_features)).reshape(n_train_next, WINDOW_SIZE, n_features)
X_test_next_scaled = next_scaler.transform(X_test_next.reshape(-1, n_features)).reshape(X_test_next.shape[0], WINDOW_SIZE, n_features)

next_class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train_next), y=y_train_next)
next_class_weight_dict = {i: weight for i, weight in enumerate(next_class_weights)}

next_model = build_model(architectures[best_name](), n_features, len(classes))
next_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
next_model.fit(
    X_train_next_scaled, y_train_next, validation_split=0.15, epochs=40, batch_size=32,
    class_weight=next_class_weight_dict, callbacks=[early_stopping()], verbose=2,
)

next_test_loss, next_test_accuracy = next_model.evaluate(X_test_next_scaled, y_test_next, verbose=0)
next_predictions_named = [index_to_label[i] for i in np.argmax(next_model.predict(X_test_next_scaled, verbose=0), axis=1)]
y_test_next_named = [index_to_label[i] for i in y_test_next]

print(f"\nFault forecaster test accuracy: {next_test_accuracy * 100:.2f}% "
      f"(vs {results[best_name]['accuracy'] * 100:.2f}% for {best_name} nowcasting on the same test split)")
print("Confusion Matrix:")
print(confusion_matrix(y_test_next_named, next_predictions_named, labels=classes))
print("Classification Report:")
print(classification_report(y_test_next_named, next_predictions_named, labels=classes))

next_model.save("models/fault_next_model.keras")
joblib.dump(next_scaler, "models/fault_next_scaler.joblib")
joblib.dump(
    {"classes": classes, "features": ALL_FEATURES, "window_size": WINDOW_SIZE, "horizon": PREDICTION_HORIZON},
    "models/fault_next_meta.joblib",
)

best_model = fine_tune_on_real_data(
    best_model, "fault", scaler, classes, label_to_index,
    X_train_scaled, y_train, results[best_name]["accuracy"], X_test_scaled, y_test,
    n_features, horizon=0,
)
next_model = fine_tune_on_real_data(
    next_model, "fault_next", next_scaler, classes, label_to_index,
    X_train_next_scaled, y_train_next, next_test_accuracy, X_test_next_scaled, y_test_next,
    n_features, horizon=PREDICTION_HORIZON,
)
