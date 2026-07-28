# UAV Flight Mode Classifier

Models that predict a UAV's flight mode from sensor readings (vertical speed, horizontal speed, roll angle, pitch angle), including a version that runs on a real PX4 flight log.

## flight_data.py

Shared constants used by both classifiers: `FEATURES`, `MODE_CYCLE`, and `MODE_PARAMS` (the per-mode sensor mean/std used to generate synthetic data).

It also holds two domain-randomization helpers, added after real PX4 logs exposed how narrow the original single-profile `MODE_PARAMS` was (see `real_log_inference.py` below):

- `SPEED_SCALE_RANGE` / a random per-flight (or, for the independent-sample RF dataset, per-row) speed-scale factor applied to the "travel" modes (`ascend`, `cruise`, `rtl`, `descend`), so training data spans both fast point-to-point missions and slow survey/loiter missions instead of only one fixed speed profile.
- `PITCH_TRIM_RANGE` / a random constant pitch offset applied to every mode, simulating airframe-specific rigging/CG trim bias that has nothing to do with flight mode.

`randomize_mode_params(params, speed_scale, pitch_trim)` applies both to a copy of `MODE_PARAMS` and is called once per generated flight (sequence classifier) or once per generated row (RF classifier, which has no flight/session concept to tie a consistent trim across rows).

## flight_mode_classifier.py

Multi-class classification on independent sensor readings (hover, takeoff, ascend, cruise, rtl, descend, land, transition).

- Dataset: Synthetic (generated with numpy, based on realistic per-mode sensor ranges, with blended transition samples between adjacent modes in the mission cycle)
- Algorithm: Random Forest Classifier

Test accuracy is around 77% (down from 84% before the pitch-trim randomization above was added). This is an inherent limitation of classifying from a single independent reading: without seeing the rest of the flight, the model has no way to tell a real mode from "that mode, offset by this airframe's trim." The sequence classifier below doesn't have this problem, because its delta (rate-of-change) features cancel out any constant trim.

Run:

```bash
python flight_mode_classifier.py
```

## flight_sequence_classifier.py

Time-series classification over sliding windows of consecutive sensor readings, with an added `anomaly` class simulating sensor malfunction or erratic control loss.

- Dataset: Synthetic continuous flight logs (120 simulated flights, each cycling through a full mission — hover, takeoff, ascend, cruise, rtl, descend, land — with smooth transitions, and occasional injected anomaly segments)
- Features: the 4 raw sensor readings plus their timestep-to-timestep delta (rate of change), 8 features total
- Algorithm: trains and compares three architectures (LSTM, GRU, and Conv1D, Keras/TensorFlow) on 10-timestep sliding windows, each with a dropout layer before the classification head
- Train/test split is done by flight, not by window, to avoid data leakage between overlapping windows
- Class weighting (`sklearn.utils.class_weight`) is used to counter the rarity of the `anomaly` class
- Flight-level 5-fold cross-validation (also split by flight, not window) is run before the final train/test split, to check that architecture ranking isn't a fluke of one particular split. LSTM is consistently the best of the three (~85% mean CV accuracy), Conv1D consistently the worst (~80%, and highest variance across folds)

Running this script also saves the best-performing model (`flight_mode_model.keras`), its feature scaler (`flight_mode_scaler.joblib`), and metadata (`flight_mode_meta.joblib`) to disk for reuse.

Run:

```bash
python flight_sequence_classifier.py
```

## real_log_inference.py

Runs the saved model on a real PX4 flight log (`.ulg` file) instead of synthetic data. Parses `vehicle_local_position` (for vertical/horizontal speed) and `vehicle_attitude` (quaternion converted to roll/pitch) with `pyulog`, builds the same sliding windows, and predicts a flight mode for each.

Output includes a per-segment summary (mode, start/end time, duration, mean confidence) printed to the console, and a saved `<log_name>_flight_mode_timeline.png` plot with the sensor traces (speeds, angles) stacked above a colored band showing the predicted mode over time.

`data/sample.ulg` is a real PX4 log (from the [pyulog](https://github.com/PX4/pyulog) test suite) used as the example input. Note: this particular log is a stationary bench/attitude test, not an actual flight (near-zero velocity, but large roll swings from the vehicle being manually rotated) — depending on the trained model, it may not predict `hover` throughout, which is a real example of the sim-to-real gap: the synthetic training data never paired large attitude changes with near-zero velocity, since normal flight doesn't produce that combination.

Two genuine flights are also included, and both drove the `MODE_PARAMS` widening in `flight_data.py`:

- `data/real_flight.ulg` — a ~10-minute flight, low horizontal speed throughout (0-3.5 m/s) and a persistent ~-12 deg pitch trim even at hover. Against the original `MODE_PARAMS` (single fixed speed profile, hover pitch centered on 0), this log broke the model in two different, instructive ways: first it predicted `transition` almost everywhere (no travel mode's speed range was ever close to this slow), and after the speed-scale fix alone, it flipped to predicting `descend` almost everywhere (its constant negative pitch trim looked exactly like the `descend` class' pitch mean once slow-mode speeds started overlapping hover). Only after also randomizing pitch trim per flight did the predicted timeline start tracking the actual altitude changes (e.g. the sustained descent in the last ~60s of the flight is correctly predicted as `descend`).
- `data/real_flight_2.ulg` — a shorter (~3.5 min), faster, near-zero-trim flight (horizontal speed up to 14 m/s). It validates the other end of the widened range: the two fast horizontal bursts early in the flight are correctly picked up as `cruise`/`rtl` instead of being invisible to the model.

Requires `flight_sequence_classifier.py` to have been run first (to produce the saved model files).

Run:

```bash
python flight_sequence_classifier.py
python real_log_inference.py data/real_flight.ulg
```

## app.py

Streamlit UI around the same inference pipeline as `real_log_inference.py`. Upload a `.ulg` file (or fall back to the bundled sample log), and it shows flight duration/sample count/mean confidence, the predicted mode distribution, the sensor + mode timeline, and a table of predicted segments.

Requires `flight_sequence_classifier.py` to have been run first (to produce the saved model files).

Run:

```bash
streamlit run app.py
```

## Setup

```bash
pip install -r requirements.txt
```
