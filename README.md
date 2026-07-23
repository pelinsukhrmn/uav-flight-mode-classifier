# UAV Flight Mode Classifier

Models that predict a UAV's flight mode from sensor readings (vertical speed, horizontal speed, roll angle, pitch angle), including a version that runs on a real PX4 flight log.

## flight_mode_classifier.py

Multi-class classification on independent sensor readings (hover, takeoff, ascend, cruise, rtl, descend, land, transition).

- Dataset: Synthetic (generated with numpy, based on realistic per-mode sensor ranges, with blended transition samples between adjacent modes in the mission cycle)
- Algorithm: Random Forest Classifier

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

Running this script also saves the best-performing model (`flight_mode_model.keras`), its feature scaler (`flight_mode_scaler.joblib`), and metadata (`flight_mode_meta.joblib`) to disk for reuse.

Run:

```bash
python flight_sequence_classifier.py
```

## real_log_inference.py

Runs the saved model on a real PX4 flight log (`.ulg` file) instead of synthetic data. Parses `vehicle_local_position` (for vertical/horizontal speed) and `vehicle_attitude` (quaternion converted to roll/pitch) with `pyulog`, builds the same sliding windows, and predicts a flight mode for each.

Output includes a per-segment summary (mode, start/end time, duration, mean confidence) printed to the console, and a saved `<log_name>_flight_mode_timeline.png` plot with the sensor traces (speeds, angles) stacked above a colored band showing the predicted mode over time.

`data/sample.ulg` is a real PX4 log (from the [pyulog](https://github.com/PX4/pyulog) test suite) used as the example input. Note: this particular log is a stationary bench/attitude test, not an actual flight (near-zero velocity, but large roll swings from the vehicle being manually rotated) — depending on the trained model, it may not predict `hover` throughout, which is a real example of the sim-to-real gap: the synthetic training data never paired large attitude changes with near-zero velocity, since normal flight doesn't produce that combination. Passing a genuine flight log (real climb/cruise/descent) as an argument would be a better test of the model's real-world accuracy.

Requires `flight_sequence_classifier.py` to have been run first (to produce the saved model files).

Run:

```bash
python flight_sequence_classifier.py
python real_log_inference.py data/sample.ulg
```

## Setup

```bash
pip install -r requirements.txt
```
