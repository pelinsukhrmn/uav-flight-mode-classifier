# UAV Flight Mode Classifier

A flight-mode classifier for PX4 drones, built to run in real time. It reads four sensor values (vertical speed, horizontal speed, roll, pitch) and outputs the current flight mode plus a ~1-2 second forecast of the next one. The model is an LSTM trained on synthetic mission data and fine-tuned on eleven real PX4 flight logs. The inference path is also ported to dependency-free C++, which runs both models in well under a millisecond combined - fast enough for a real control loop, not just a demo.

Read in this order: [flight_sequence_classifier.py](#flight_sequence_classifierpy) (train the model) → [real_log_inference.py](#real_log_inferencepy) (check it against real flights) → [live_inference.py](#live_inferencepy) and [cpp/](#cpp) (run it in real time) → [app.py](#apppy) (a Streamlit UI for poking at all of the above). [flight_mode_classifier.py](#baseline-flight_mode_classifierpy-superseded) is an earlier, superseded approach kept for comparison.

Live demo: https://pelinsukhrmn-uav-flight-mode-classifier-app-u3jseo.streamlit.app

## Setup

```bash
pip install -r requirements.txt
```

## flight_data.py

Shared constants for both classifiers: `FEATURES`, `MODE_CYCLE`, and `MODE_PARAMS` (per-mode sensor mean/std used to generate synthetic training data).

Two randomization helpers were added after real PX4 logs showed the original single-profile `MODE_PARAMS` was too narrow:

- `SPEED_SCALE_RANGE` - a random per-flight speed multiplier applied to the "travel" modes (`ascend`, `cruise`, `rtl`, `descend`), so training data covers both fast point-to-point missions and slow survey flights.
- `PITCH_TRIM_RANGE` - a random constant pitch offset per flight, simulating airframe rigging/CG trim.

`randomize_mode_params()` applies both to a copy of `MODE_PARAMS`, once per generated flight (sequence classifier) or once per row (RF baseline, which has no flight concept).

## flight_sequence_classifier.py

Trains the model. Time-series classification over sliding windows of sensor readings, with an `anomaly` class for sensor malfunction or erratic control loss.

- Dataset: 120 synthetic flights, each cycling through hover → takeoff → ascend → cruise → rtl → descend → land, with blended transitions and occasional injected anomalies.
- Features: 4 raw sensor readings plus their timestep deltas, 8 total.
- Compares four architectures (LSTM, GRU, Conv1D, BiLSTM) on 15-step sliding windows. Each recurrent layer uses 64 units with `recurrent_dropout=0.1`; Conv1D uses two 64-filter causal layers with a dropout between them. All four share a `Dropout(0.2)` + `Dense(32, relu)` head.
- Split by flight, not by window, so overlapping windows can't leak across train/test.
- Class-weighted loss for the rare `anomaly` class; `EarlyStopping` on validation accuracy for every CV fold and the final run.
- 5-fold flight-level CV runs before the final split, to check the architecture ranking isn't a fluke of one split.

Results:

| Architecture | Test accuracy | 5-fold CV |
|---|---|---|
| BiLSTM | 86.89% | 86.32% ± 0.69% |
| LSTM | 86.03% | 85.95% ± 0.73% |
| GRU | 84.30% | 85.78% ± 0.80% |
| Conv1D | 80.59% | 79.72% ± 1.48% |

`transition` is the hardest class everywhere (LSTM recall 0.52, vs 0.96+ for every other class) - the synthetic transition samples are blended between two real classes by construction, so some confusion is expected.

BiLSTM usually wins the comparison, but production is **pinned to LSTM** regardless: `cpp/lstm_model.hpp` only implements a single-layer unidirectional LSTM, and shipping a BiLSTM would break C++ parity.

The pinned LSTM is then fine-tuned on real data (see [Real-data fine-tuning](#real-data-fine-tuning)) before shipping. Running this script saves `models/flight_mode_model.keras`, its scaler, and metadata, plus a second forecasting model.

Run:

```bash
python src/flight_sequence_classifier.py
```

### Next-mode forecasting

The same script also trains a second model that forecasts the mode 10 steps ahead (`PREDICTION_HORIZON = 10`, roughly 1-2 seconds depending on sample rate) from the same 15-step window - only the label shifts. It reuses whichever architecture won above rather than repeating the full comparison.

- Saved as `models/flight_mode_next_model.keras` / `..._scaler.joblib` / `..._meta.joblib`.
- Test accuracy: 60.12%, vs 86.03% for nowcasting on the same split. Forecasting ahead is a harder problem, especially near transitions - by the time a transition is 10 steps from ending, the model tends to already commit to the arriving mode instead of predicting `transition`.
- Unlike the nowcaster, **the forecaster isn't part of the real-data fine-tuning pass** - it's synthetic-only. See `real_flight_11_descend.ulg`'s 5.1% score below for what that costs on a class it's never seen real examples of.

Measured on real logs, where ground truth exists 10 steps ahead:

| Log | Next-mode accuracy | Coverage |
|---|---|---|
| `real_flight.ulg` | 65.8% | 4.9% (149 windows) |
| `real_flight_2.ulg` | 49.2% | 36.4% (815 windows) |
| `real_flight_3_vtol.ulg` | 0.0% | 18.6% (3662 windows) |
| `real_flight_6_takeoff_land.ulg` | 71.2% | 100.0% (3717 windows) |
| `real_flight_7_takeoff_land.ulg` | 2.4% | 100.0% (2269 windows) |
| `real_flight_8_hover_rtl.ulg` | 95.8% | 75.8% (3846 windows) |
| `real_flight_9_hover_land.ulg` | 89.6% | 96.3% (6023 windows) |
| `real_flight_10_sitl_hover_rtl.ulg` | 66.9% | 60.7% (11934 windows) |
| `real_flight_11_descend.ulg` | 5.1% | 100.0% (3354 windows) |

Two of these need more than the number:

- `real_flight_11_descend.ulg` (5.1%): on its 2616 true-`descend` windows, the forecaster predicts `cruise` 68% of the time and `anomaly` 26% - almost never `descend`. It's working from synthetic `descend`/`cruise` boundaries only, and they don't hold up on this log's real vertical-speed profile 10 steps out.
- `real_flight_7_takeoff_land.ulg` (2.4%): PX4's `nav_state` stays in `AUTO_TAKEOFF` for most of this flight even though the vehicle is sitting still at altitude for over 3 minutes. The forecaster predicts `hover` for 97% of the true-`takeoff` windows - the physically correct read - but it's scored against a `takeoff` label that doesn't reflect what the aircraft is actually doing. A label-quality issue in this specific log, not a model failure.

### Real-data fine-tuning

The CV-winning model gets a fine-tuning pass on ground-truth-labeled real windows, gated so real data can only improve the production model, never quietly regress it.

- Logs: `real_flight.ulg`, `real_flight_2.ulg`, four public logs from [PX4's Flight Review database](https://review.px4.io) (`real_flight_6` through `real_flight_9`), a fifth picked for `descend` coverage (`real_flight_11`), and a PX4 SITL+Gazebo flight (`real_flight_10`). `real_flight_3_vtol`, `_4_poshold`, and `_5_stab` stay evaluation-only.
- Each log's covered windows are split temporally 70/30 (chronological, not shuffled, so overlapping windows can't leak across the split) - 70% fine-tune, 30% held out.
- Fine-tuning batch = real training windows plus a stratified synthetic "replay" sample (up to 3x the real count), so the four classes with no real coverage (`ascend`/`cruise`/`transition`/`anomaly`) aren't forgotten.
- Low learning rate (1e-4), up to 15 epochs, early stopping on real held-out accuracy.
- **Acceptance test**: the fine-tuned model only ships if it matches or beats the real held-out accuracy and doesn't lose more than 3 points of synthetic test accuracy. Otherwise the synthetic-only model stays in production.
- Pinned to LSTM, same C++-parity reason as above.

Current result (8 logs, 22,506 fine-tune-train windows):

| | Real held-out accuracy | Synthetic test accuracy |
|---|---|---|
| Before fine-tuning | 40.43% (9651 windows) | 86.03% |
| After fine-tuning | 55.37% (+14.94) | 86.58% (+0.55) |

Both numbers moved up together this round - earlier, smaller-capacity versions of this model sometimes traded a few points of synthetic accuracy for real-world gains; this one had enough headroom to improve both at once.

One thing worth flagging: `real_flight_10_sitl_hover_rtl.ulg` alone supplies 8360 of the 22,506 fine-tune windows (37%), more than any single hardware log - a ~157s SITL flight at PX4's native sample rate just produces far more windows than a similarly short public log. It's simulated (clean Gazebo physics, no sensor noise), not hardware, so its weight in the mix matters; `app.py`'s dropdown marks it "simulated" explicitly.

## real_log_inference.py

Runs the saved model on a real `.ulg` log instead of synthetic data: parses `vehicle_local_position` and `vehicle_attitude` with `pyulog`, builds the same windows, predicts a mode for each.

Output: a per-segment summary (mode, start/end, duration, mean confidence), a ground-truth accuracy line where available, and a `<log>_flight_mode_timeline.png` plot.

### Ground truth, not just eyeballing the timeline

PX4 logs carry `vehicle_status.nav_state`, the autopilot's own record of flight mode. `flight_mode_inference.load_ground_truth` / `evaluate_predictions` turn that into an accuracy number, over the subset of `nav_state` values that map unambiguously onto our labels:

| `nav_state` | Our label |
|---|---|
| `AUTO_LOITER` | `hover` |
| `AUTO_TAKEOFF`, `AUTO_VTOL_TAKEOFF` | `takeoff` |
| `AUTO_LAND`, `AUTO_PRECLAND` | `land` |
| `AUTO_RTL` | `rtl` |
| `DESCEND` | `descend` |

`AUTO_MISSION` and the manual modes (`MANUAL`, `ALTCTL`, `POSCTL`, `STAB`, `ACRO`, `OFFBOARD`) are left unmapped - `AUTO_MISSION` alone could be any of `ascend`/`cruise`/`descend`/`hover` depending on the leg, and the manual modes have no equivalent label at all. `coverage` is how much of the flight fell into a mapped state; accuracy is only computed over that portion.

Measured with the current model (LSTM, fine-tuned on real data):

| Log | Ground-truth accuracy | Coverage |
|---|---|---|
| `sample.ulg` | n/a (never leaves `MANUAL`) | 0% |
| `real_flight.ulg` | 67.9% | 5.2% (159 windows) |
| `real_flight_2.ulg` | 76.6% | 36.2% (815 windows) |
| `real_flight_3_vtol.ulg` | 0.0% | 18.6% (3662 windows) |
| `real_flight_4_poshold.ulg` | n/a (all `POSCTL`) | 0% |
| `real_flight_5_stab.ulg` | n/a (all `STAB`) | 0% |
| `real_flight_6_takeoff_land.ulg` | 87.2% | 100.0% (3727 windows) |
| `real_flight_7_takeoff_land.ulg` | 96.1% | 100.0% (2279 windows) |
| `real_flight_8_hover_rtl.ulg` | 96.0% | 75.6% (3846 windows) |
| `real_flight_9_hover_land.ulg` | 88.8% | 96.1% (6023 windows) |
| `real_flight_10_sitl_hover_rtl.ulg` | 70.5% | 60.7% (11944 windows) |
| `real_flight_11_descend.ulg` | 93.4% | 100.0% (3364 windows) |

`real_flight_3_vtol.ulg` is a genuine domain gap. During its `AUTO_LOITER`/`AUTO_RTL` segments the aircraft is circling in fixed-wing mode at 16-18 m/s with 27-33° of bank - nothing like the near-stationary multirotor `hover`/`rtl` the model trained on. On its true-`hover` windows it now predicts `transition` 72% of the time and `cruise` 27%, never `hover`. Which wrong label it lands on has shifted across retrains (it's picked `anomaly`, a `hover`/`cruise` split, and now `transition` in different rounds) - that's noise on an input the model was never trained to handle, not a trend worth reading into.

The public multirotor logs (`real_flight_6` through `real_flight_9`, `real_flight_11`) score 87-96%, close to what the fine-tuning set is dominated by. `real_flight_10` (70.5%) is simulated, not hardware - see [Real-data fine-tuning](#real-data-fine-tuning) for its outsized weight in the training mix.

`data/sample.ulg` (from [pyulog](https://github.com/PX4/pyulog)'s test suite) is a stationary bench test, not a flight - near-zero velocity with large roll swings from manual handling. The synthetic training data never pairs large attitude changes with near-zero velocity, so this is a clean sim-to-real gap example. No ground truth (`nav_state` never leaves `MANUAL`).

Eleven real flights are included:

- `real_flight.ulg` - ~10 min multirotor, slow (0-3.5 m/s) with a persistent ~-12° pitch trim at hover. This log is why `flight_data.py` randomizes speed scale and pitch trim per flight - without that, the original fixed-profile model predicted `transition` almost everywhere, then `descend` almost everywhere.
- `real_flight_2.ulg` - ~3.5 min, faster (up to 14 m/s), near-zero trim. Validates the fast end of the speed range: two horizontal bursts are correctly read as `cruise`/`rtl`.
- `real_flight_3_vtol.ulg` - ~33 min VTOL test flight, mostly manual `STAB`/`ACRO` with brief `AUTO_LOITER`/`AUTO_RTL`/`AUTO_MISSION` segments. A different vehicle domain entirely, included as a stress test rather than a case the model is expected to handle.
- `real_flight_4_poshold.ulg` - ~2.4 min, entirely `POSCTL` (manual position hold). No ground truth by design; a real hand-flown example for the timeline.
- `real_flight_5_stab.ulg` - ~1 min, entirely `STAB`. Same story.
- `real_flight_6_takeoff_land.ulg` through `real_flight_9_hover_land.ulg` - four public quadrotor logs from [PX4's Flight Review database](https://review.px4.io), picked from ~1300 candidates for a good community rating, zero logged errors, and strong `hover`/`takeoff`/`land`/`rtl` coverage. Different airframes than `real_flight.ulg`/`real_flight_2.ulg`, on purpose.
- `real_flight_10_sitl_hover_rtl.ulg` - ~157s, flown entirely in PX4 SITL + Gazebo (`gz_x500`): a scripted MAVSDK mission (arm → 4-waypoint mission → paused for an `AUTO_LOITER` hold → resumed → `AUTO_RTL`). Ground truth covers `hover` (140 windows) and `rtl` (62). Simulated, not hardware - labeled as such in `app.py`.
- `real_flight_11_descend.ulg` - public quadrotor log picked for `DESCEND` nav_state coverage (530 of 690 ground-truth windows), the first real source for the `descend` class.

Requires `flight_sequence_classifier.py` to have been run first.

Run:

```bash
python src/flight_sequence_classifier.py
python src/real_log_inference.py data/real_flight.ulg
```

## live_inference.py

A streaming version of the same pipeline: feeds feature rows in one at a time (replayed log or live MAVLink) instead of loading a whole file, maintains a rolling window via `LiveWindowBuffer` (reuses `flight_mode_inference.build_windows`, so streaming and offline preprocessing can't drift apart), and scores both models as each window completes. Advisory only - it never writes back to the vehicle.

Two sources:

- `replay_log_source` - replays a `.ulg` file's real timestamps (or as fast as possible with `--speed 0`). Exercised end to end: `python src/live_inference.py --mode replay --log data/sample.ulg --speed 0 --min-interval 1.0`.
- `mavlink_source` - connects to a live MAVLink stream via `pymavlink`, reading `ATTITUDE`/`LOCAL_POSITION_NED`. Written against the documented fields but not exercised here - no SITL toolchain or live vehicle available.

`--min-interval` (default 1.0s) throttles scoring - MAVLink messages can arrive tens of times a second, and mode changes don't need sub-second resolution.

Run:

```bash
python src/live_inference.py --mode replay --log data/sample.ulg --speed 0
python src/live_inference.py --mode mavlink --connection udp:127.0.0.1:14540  # untested here
```

## cpp/

A dependency-free C++ port of the inference path. Python's `model.predict()` costs tens of milliseconds per call regardless of batch size - fine for a demo, not for a control loop. No TFLite/ONNXRuntime: this machine only has MinGW g++, and those runtimes' prebuilt Windows binaries target MSVC's ABI. The model is small enough (~21k params: single-layer LSTM(64) + Dense(32) + Dense(9)) that a hand-written forward pass is simpler to build and faster to run than pulling in a general-purpose runtime.

- `export_weights.py` - dumps trained weights + scaler into `weights_current_mode.h` / `weights_next_mode.h`. Regenerate after every retrain.
- `lstm_model.hpp` - the forward pass: standardize → single-layer LSTM (Keras gate order) → Dense+ReLU → Dense+softmax. One implementation, two models, via a `Weights` struct of pointers.
- `verify_parity.py` + `parity_check.cpp` - confirms the C++ forward pass matches Keras. Checked on 200 real windows (100 current-mode + 100 next-mode, from `real_flight_2.ulg`): 0 label mismatches, max probability difference 0.000001.
- `benchmark.cpp` - isolated latency measurement. Both models combined: **~460 microseconds per prediction** on this dev machine - comfortably inside a 100Hz (10ms) control-loop budget, but noticeably higher than an earlier, smaller version of this model (~4 microseconds at 32 units instead of 64). The likely cause is cache locality, not raw compute: `lstm_recurrent_kernel` grew from 16KB to 64KB between the two, and the hand-written loop's access pattern stops being cache-friendly around that size. Not fixed yet - see [Future work](#future-work).
- `export_replay_csv.py` + `main_replay.cpp` - the C++ equivalent of `live_inference.py --mode replay`: streams a log from CSV, buffers a window, scores both models, prints the same advisory lines.
- No MAVLink ingestion in C++ - not worth vendoring the headers and standing up a SITL connection just to leave it untested, same reasoning as the Python side.

`weights_current_mode.h`/`weights_next_mode.h` are generated output that can drift from the actual `.keras` models if someone retrains and forgets to re-export. The `cpp-parity` job in `.github/workflows/tests.yml` catches this on every push: it regenerates the headers and fails the build if they don't match, then rebuilds and reruns the parity check against the live models.

Build and run (MinGW g++, no other dependencies):

```bash
python cpp/export_weights.py
g++ -std=c++17 -O2 -o cpp/parity_check.exe cpp/parity_check.cpp && python cpp/verify_parity.py && ./cpp/parity_check.exe
g++ -std=c++17 -O2 -o cpp/benchmark.exe cpp/benchmark.cpp && ./cpp/benchmark.exe
g++ -std=c++17 -O2 -o cpp/main_replay.exe cpp/main_replay.cpp
python cpp/export_replay_csv.py data/real_flight_2.ulg && ./cpp/main_replay.exe cpp/replay_data.csv 1.0
```

## app.py

A Streamlit UI for inspecting the pipeline without a MAVLink connection or a build step - not the deployment target, just the fastest way to eyeball a flight. Upload a `.ulg` file or pick one of the twelve bundled logs. Shows duration/sample count/mean confidence, ground-truth accuracy where available, mode distribution, a sensor+mode timeline, and a segment table with CSV download. Also shows a next-mode forecast section when the forecaster artifacts exist.

Requires `flight_sequence_classifier.py` to have been run first.

Run:

```bash
streamlit run src/app.py
```

## Baseline: flight_mode_classifier.py (superseded)

The first approach tried here, kept for comparison. Classifies **independent** sensor readings - no window, no memory of the rest of the flight - into the same 8 modes plus `transition`.

- Dataset: synthetic, same per-mode ranges as above.
- Algorithm: Random Forest.

Test accuracy: ~77% (down from 84% before pitch-trim randomization was added - a single reading has no way to distinguish a real mode from "that mode, offset by this airframe's trim"). This is exactly why the project moved to the sequence model, whose delta features cancel out any constant trim.

Run:

```bash
python src/flight_mode_classifier.py
```

## Tests

Unit tests for the deterministic pieces: domain randomization, quaternion conversion, nav_state mapping, windowing (including the horizon shift), `evaluate_predictions`, `temporal_split`, artifact loading, segment summarization, and `live_inference.py`'s buffer/advisory logic. No `.ulg` files or trained model required. Runs on push/PR.

```bash
pytest
```

## Future work

- **C++ inference latency regressed with the larger model** (~4us → ~460us per prediction, both models combined - see [cpp/](#cpp)). Likely a cache-locality issue in `lstm_model.hpp`'s loop order at 64 units, not the extra compute itself. Still well inside a real control-loop budget, but worth fixing before trusting this number at a larger model size.
- **MAVLink is untested end to end.** `live_inference.py`'s `mavlink_source` is written against the documented message fields but never run against a live SITL instance or radio link. No MAVLink ingestion in `cpp/` at all.
- **The ~460us number is from this dev machine**, not a companion computer (Raspberry Pi, Jetson) - unmeasured there, and doesn't include MAVLink parsing/buffering overhead.
- **`ascend`/`cruise` have no real ground truth.** They only appear as sub-legs of `AUTO_MISSION`, which PX4 doesn't record as a distinct `nav_state`. Real labels would need inferring mission legs from waypoint/altitude-setpoint data.
- **`anomaly` is 100% synthetic** - the one class where being wrong matters most. PX4's Flight Review database has logs with genuine faults (logged errors, vibration, sensor-error tags), but turning "this log has errors somewhere" into per-window ground truth needs its own labeling approach; `nav_state` has no `anomaly` state to map from.
- **The forecaster doesn't get the real-data fine-tuning pass** the nowcaster does - see `real_flight_11_descend.ulg`'s 5.1% forecast score for what that costs.
- `live_inference.py` and `cpp/` are advisory-only by design, not wired into the autopilot. Closing that loop needs real `anomaly` ground truth, a measured latency budget on actual target hardware, and a case for why a probabilistic classifier should override PX4's own mode state machine at all.
