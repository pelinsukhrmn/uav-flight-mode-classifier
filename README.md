# UAV Flight Mode Classifier

Models that predict a UAV's flight mode from sensor readings (vertical speed, horizontal speed, roll angle, pitch angle), including a version that runs on a real PX4 flight log.

Live demo: https://pelinsukhrmn-uav-flight-mode-classifier-app-u3jseo.streamlit.app

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
- Algorithm: trains and compares four architectures (LSTM, GRU, Conv1D, and BiLSTM, Keras/TensorFlow) on 10-timestep sliding windows, each with a dropout layer before the classification head. BiLSTM is safe to add here (unlike a real-time streaming model) because each window is already a fixed, fully-past slice by the time it's classified — the "future" a backward pass sees is still only within that window, not beyond the current prediction point.
- Train/test split is done by flight, not by window, to avoid data leakage between overlapping windows
- Class weighting (`sklearn.utils.class_weight`) is used to counter the rarity of the `anomaly` class
- `EarlyStopping` (`monitor="val_accuracy"`, `patience=3`, `restore_best_weights=True`) is used for both the CV folds and the final training run, so each architecture is compared/saved at its own best epoch instead of a fixed epoch count that may over- or under-train it
- Flight-level 5-fold cross-validation (also split by flight, not window) is run before the final train/test split, to check that architecture ranking isn't a fluke of one particular split

Latest run's held-out test accuracy: **BiLSTM 88.70%** (best, now the saved model) > LSTM 86.83% > GRU 86.37% > Conv1D 84.32% (still the weakest, as before). `transition` is the hardest class for every architecture by a wide margin (recall 0.46-0.63 vs 0.88+ for every other class) — the blended synthetic transition samples straddle two real classes by construction, so some confusion there is expected.

**BiLSTM being the best on synthetic test accuracy did not carry over to the real logs** — see the ground-truth results below, where it's actually very slightly worse than the old LSTM model on 2 of the 3 logs with ground-truth coverage. A few extra points of synthetic test accuracy is not the same thing as better real-world generalization; it's a genuine reminder of why this project validates against real PX4 logs at all instead of stopping at the synthetic test split.

Running this script also saves the best-performing model (`flight_mode_model.keras`), its feature scaler (`flight_mode_scaler.joblib`), and metadata (`flight_mode_meta.joblib`) to disk for reuse.

Run:

```bash
python flight_sequence_classifier.py
```

## real_log_inference.py

Runs the saved model on a real PX4 flight log (`.ulg` file) instead of synthetic data. Parses `vehicle_local_position` (for vertical/horizontal speed) and `vehicle_attitude` (quaternion converted to roll/pitch) with `pyulog`, builds the same sliding windows, and predicts a flight mode for each.

Output includes a per-segment summary (mode, start/end time, duration, mean confidence) printed to the console, a ground-truth accuracy line where available (see below), and a saved `<log_name>_flight_mode_timeline.png` plot with the sensor traces (speeds, angles) stacked above a colored band showing the predicted mode over time.

### Ground-truth accuracy, not just eyeballing the timeline

Real PX4 logs also carry `vehicle_status.nav_state` — PX4's own record of which flight mode it was in. Previous versions of this project could only judge real-log predictions by eye ("the descent in the timeline lines up with the altitude drop"). `flight_mode_inference.load_ground_truth` / `evaluate_predictions` turn that into a real accuracy number, but only over the subset of `nav_state` values that map unambiguously onto our labels:

| `nav_state` | Our label |
|---|---|
| `AUTO_LOITER` | `hover` |
| `AUTO_TAKEOFF`, `AUTO_VTOL_TAKEOFF` | `takeoff` |
| `AUTO_LAND`, `AUTO_PRECLAND` | `land` |
| `AUTO_RTL` | `rtl` |

Everything else (`MANUAL`, `ALTCTL`, `POSCTL`, `STAB`, `ACRO`, `OFFBOARD`, `AUTO_MISSION`, ...) is left unmapped on purpose — `AUTO_MISSION` alone could be `ascend`/`cruise`/`descend`/`hover` depending on which leg of the mission it is, and the manual modes have no equivalent in our label set at all. The reported `coverage` is how much of the flight fell into a mapped state; accuracy is only computed over that covered portion, so a low-coverage flight (e.g. one flown mostly by hand) is reported honestly as "not much to validate here" rather than silently padded with guesses.

Measured with the current (BiLSTM) model:

| Log | Ground-truth accuracy | Coverage |
|---|---|---|
| `sample.ulg` | n/a (never leaves `MANUAL`) | 0% |
| `real_flight.ulg` | 59.1% | 5.4% |
| `real_flight_2.ulg` | 69.0% | 36.1% |
| `real_flight_3_vtol.ulg` | 0.0% | 18.6% |
| `real_flight_4_poshold.ulg` | n/a (all `POSCTL`) | 0% |
| `real_flight_5_stab.ulg` | n/a (all `STAB`) | 0% |

`real_flight_3_vtol.ulg`'s 0% is a genuine, verified domain-gap result, not a measurement artifact: during its `AUTO_LOITER`/`AUTO_RTL` segments the aircraft is actually circling in fixed-wing mode at ~16-18 m/s horizontal speed with 27-33° of bank — nothing like the near-stationary, near-level multirotor `hover`/`rtl` the model was trained on. It mostly predicts `anomaly` there instead, which is arguably the "least wrong" label available to it, since the closest true label (`hover`/`rtl`) isn't in its vocabulary for that flight regime.

`real_flight.ulg` and `real_flight_2.ulg` also dropped a few points versus the previous (LSTM) model (was 65.9%/70.4%) — a small, direct illustration of the point above: BiLSTM's higher synthetic test accuracy didn't translate to better real-world accuracy on these two logs.

`data/sample.ulg` is a real PX4 log (from the [pyulog](https://github.com/PX4/pyulog) test suite) used as the example input. Note: this particular log is a stationary bench/attitude test, not an actual flight (near-zero velocity, but large roll swings from the vehicle being manually rotated) — depending on the trained model, it may not predict `hover` throughout, which is a real example of the sim-to-real gap: the synthetic training data never paired large attitude changes with near-zero velocity, since normal flight doesn't produce that combination. Its `nav_state` never leaves `MANUAL`, so it has no ground-truth coverage at all.

Five genuine flights are also included:

- `data/real_flight.ulg` — a ~10-minute multirotor flight, low horizontal speed throughout (0-3.5 m/s) and a persistent ~-12 deg pitch trim even at hover. Against the original `MODE_PARAMS` (single fixed speed profile, hover pitch centered on 0), this log broke the model in two different, instructive ways: first it predicted `transition` almost everywhere (no travel mode's speed range was ever close to this slow), and after the speed-scale fix alone, it flipped to predicting `descend` almost everywhere (its constant negative pitch trim looked exactly like the `descend` class' pitch mean once slow-mode speeds started overlapping hover). Only after also randomizing pitch trim per flight did the predicted timeline start tracking the actual altitude changes (e.g. the sustained descent in the last ~60s of the flight is correctly predicted as `descend`). Ground truth covers its `AUTO_LOITER`/`AUTO_TAKEOFF`/`AUTO_LAND` segments.
- `data/real_flight_2.ulg` — a shorter (~3.5 min), faster, near-zero-trim multirotor flight (horizontal speed up to 14 m/s). It validates the other end of the widened range: the two fast horizontal bursts early in the flight are correctly picked up as `cruise`/`rtl` instead of being invisible to the model.
- `data/real_flight_3_vtol.ulg` — a ~33-minute VTOL (fixed-wing-capable) test flight, flown mostly in manual `STAB`/`ACRO` modes with only brief `AUTO_LOITER`/`AUTO_RTL`/`AUTO_MISSION` segments. This is a different domain-gap example from the other two: the model was only ever trained on a multirotor hover→takeoff→ascend→cruise→rtl→descend→land mission cycle, so a mostly hand-flown VTOL log is expected to have low ground-truth coverage and is included as an honest stress test, not a validated success case.
- `data/real_flight_4_poshold.ulg` — a ~2.4-minute multirotor flight flown entirely in `POSCTL` (manual position hold). No ground-truth coverage by design (`POSCTL` isn't in the mapping table above — a pilot can translate freely in position hold, so it has no single equivalent label), but it's a real, continuous manual-flight example for judging the timeline by eye.
- `data/real_flight_5_stab.ulg` — a ~1-minute multirotor flight flown entirely in `STAB` (manual stabilized). Same story as above: no ground-truth coverage, included as another real hand-flown segment.

Requires `flight_sequence_classifier.py` to have been run first (to produce the saved model files).

Run:

```bash
python flight_sequence_classifier.py
python real_log_inference.py data/real_flight.ulg
```

## app.py

Streamlit UI around the same inference pipeline as `real_log_inference.py`. Upload a `.ulg` file, or pick one of the six bundled logs (sample bench test, and the five real flights described above) from the dropdown. Shows flight duration/sample count/mean confidence, a ground-truth accuracy metric when the log has covered `nav_state` segments, the predicted mode distribution, the sensor + mode timeline, and a table of predicted segments with a CSV download button.

Requires `flight_sequence_classifier.py` to have been run first (to produce the saved model files).

Run:

```bash
streamlit run app.py
```

## Tests

`tests/` has unit tests for the pure/deterministic pieces of the pipeline (domain randomization, quaternion conversion, nav_state ground-truth mapping, windowing, segment summarization) — no `.ulg` files or trained model required. Runs on push/PR via `.github/workflows/tests.yml`.

```bash
pytest
```

## Setup

```bash
pip install -r requirements.txt
```
