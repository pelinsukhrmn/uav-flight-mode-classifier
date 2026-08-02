# UAV Flight Mode Classifier

Models that predict a UAV's flight mode from sensor readings (vertical speed, horizontal speed, roll angle, pitch angle), including a version that runs on a real PX4 flight log.

Live demo: https://pelinsukhrmn-uav-flight-mode-classifier-app-u3jseo.streamlit.app

## Setup

```bash
pip install -r requirements.txt
```

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
python src/flight_mode_classifier.py
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

Latest run's held-out test accuracy: **LSTM 85.08%** (best) > BiLSTM 84.90% > GRU 84.03% > Conv1D 79.80% (still the weakest, as before). Flight-level 5-fold CV ranked them differently: BiLSTM 84.91% ± 0.84% > GRU 83.75% ± 0.86% > LSTM 83.43% ± 0.87% > Conv1D 79.75% ± 1.39%. `transition` is the hardest class for every architecture by a wide margin (recall 0.44-0.47 vs 0.85+ for every other class) — the blended synthetic transition samples straddle two real classes by construction, so some confusion there is expected.

LSTM edging out BiLSTM here (85.08% vs 84.90%, 0.18 points) despite BiLSTM having the higher CV mean is a good illustration of single-split noise: the two architectures are statistically indistinguishable on this data (both comfortably inside each other's CV standard deviation), and the model selection code picks by final-test accuracy rather than CV mean, so which one "wins" and gets saved as production can flip between runs on the same data. This is expected variance, not a regression — worth knowing before reading too much into small architecture-ranking swings between README revisions.

(These numbers aren't directly comparable to revisions before the one that fixed a bug where the `architectures` dict held live Keras layer *instances* shared across every CV fold and the final training run, instead of building fresh layers each time — meaning "independent" CV folds were silently warm-starting on each other's trained weights. Fixed by converting `architectures` to a dict of factory functions, called fresh at each use site.)

Whichever architecture wins is then fine-tuned on real data (see [Real-data fine-tuning](#real-data-fine-tuning) below) before being saved as the production model — see the updated ground-truth results.

Running this script also saves the best-performing model (`models/flight_mode_model.keras`, possibly the fine-tuned version — see below), its feature scaler (`models/flight_mode_scaler.joblib`), and metadata (`models/flight_mode_meta.joblib`) to disk for reuse, plus a second forecasting model (see [Next-mode forecasting](#next-mode-forecasting) below).

Run:

```bash
python src/flight_sequence_classifier.py
```

### Next-mode forecasting

Besides predicting the *current* flight mode, this script also trains a second model that forecasts the mode `PREDICTION_HORIZON = 10` timesteps ahead (about 1-2 seconds, depending on the log's sample rate), using the exact same 10-step sliding window as input — only the label changes, from `start + window_size - 1` (current mode) to `start + window_size - 1 + horizon` (mode 10 steps later). It reuses whichever architecture won the final test-set comparison above (currently LSTM — see the note on CV-vs-final-test ranking noise there) rather than repeating the full 4-way comparison for a second task.

- Saved artifacts: `models/flight_mode_next_model.keras`, `models/flight_mode_next_scaler.joblib`, `models/flight_mode_next_meta.joblib` (metadata includes `horizon`).
- Test accuracy: **56.74%**, vs 85.08% for nowcasting the current mode on the same split. This gap is expected and reported honestly rather than hidden — forecasting ahead is a strictly harder problem than reporting what's already visible in the window, especially near mode transitions. Notably, `transition` itself gets ~0% recall in the forecaster: by the time a transition is 10 steps from completing, the model tends to already commit to whichever mode is arriving next rather than predicting "transition," since that label describes *now*, not 10 steps from now.
- `real_log_inference.py` and `app.py` both run this second model too, reporting a forecasted next mode/confidence and, where ground truth exists that far ahead, a horizon-shifted ground-truth accuracy via `evaluate_predictions(..., horizon=...)`.

Measured on the real logs (only where ground truth exists 10 steps ahead of a window):

| Log | Next-mode accuracy | Coverage |
|---|---|---|
| `real_flight.ulg` | 63.6% | 5.1% (154 windows) |
| `real_flight_2.ulg` | 67.7% | 36.3% (815 windows) |
| `real_flight_3_vtol.ulg` | 0.3% | 18.6% (3662 windows) |
| `real_flight_6_takeoff_land.ulg` | 79.2% | 100.0% (3722 windows) |
| `real_flight_7_takeoff_land.ulg` | 0.0% | 100.0% (2274 windows) |
| `real_flight_8_hover_rtl.ulg` | 96.2% | 75.7% (3846 windows) |
| `real_flight_9_hover_land.ulg` | 97.9% | 96.2% (6023 windows) |

(Same domain-gap caveat as the current-mode table below applies to the VTOL log's near-0% score.)

`real_flight_7_takeoff_land.ulg`'s 0.0% forecast score is a real, checked result, not a bug — and a different failure mode than the VTOL domain gap. For most of this flight PX4's `nav_state` stays in `AUTO_TAKEOFF` (it never transitions to `AUTO_LOITER`) even though the vehicle is physically just sitting still at altitude for over 3 minutes. The forecaster reads the physics correctly and predicts `hover` for essentially the entire flight (2284/2284 windows) — which is the right call for what the sensors show — but `nav_state` ground truth says `takeoff` throughout, so it's scored as 0% wrong against a label that itself doesn't reflect what the aircraft is actually doing. The current-mode (nowcasting) model shows the same pattern less starkly (93.8% ground-truth accuracy, because a chunk of its `takeoff` predictions do land on genuine climb segments) but its predicted mode is also mostly `hover`/`takeoff` for the stationary stretch. This is a label-quality artifact of this specific log's mission configuration, not a modeling error — worth knowing if using this log as a benchmark.

### Real-data fine-tuning

Ground-truth-labeled real windows are no longer just an evaluation set — the CV-winning synthetic-trained model now gets a fine-tuning pass on them, gated by an acceptance test so real data can only improve the production model, never silently regress it:

- `real_flight.ulg`, `real_flight_2.ulg`, and four public logs pulled from [PX4's Flight Review database](https://review.px4.io) (`real_flight_6_takeoff_land.ulg` through `real_flight_9_hover_land.ulg` — see below) are used (`real_flight_3_vtol.ulg`, `real_flight_4_poshold.ulg`, and `real_flight_5_stab.ulg` stay evaluation-only: different vehicle domain or no ground-truth coverage at all).
- Each log's ground-truth-covered windows are split **temporally** 70/30 (chronological, not shuffled — avoids near-duplicate overlapping windows leaking across the split): first 70% → fine-tune-train, last 30% → held out for evaluation.
- Fine-tuning batch = real fine-tune-train windows + a random stratified "replay" subsample of the original synthetic training set (up to 3x the real count) mixed in. This rehearsal keeps the 5 classes with zero real ground-truth coverage (`ascend`/`cruise`/`descend`/`transition`/`anomaly`) from being forgotten while the model adapts to the 4 real-covered classes (`hover`/`takeoff`/`land`/`rtl`).
- Low learning rate (`1e-4`), at most 15 epochs, `EarlyStopping` monitoring accuracy on the real held-out split.
- **Acceptance test**: the fine-tuned model only replaces `models/flight_mode_model.keras` if it (a) matches or beats the pre-fine-tune model on the real held-out set, and (b) doesn't regress synthetic test accuracy by more than 3 points. Otherwise the synthetic-only model is kept, and the script says so explicitly on the console — a claim of "improved with real data" is never made without a measured result behind it.

Result from the current fine-tuning run (6 logs — `real_flight.ulg`, `real_flight_2.ulg`, and the four public logs from PX4's Flight Review database — 11,802 ground-truth fine-tune-train windows, held-out set covering `hover`/`land`/`rtl`/`takeoff`):

| | Real held-out accuracy | Synthetic test accuracy |
|---|---|---|
| Before fine-tuning | 73.63% (5062 windows) | 85.08% |
| After fine-tuning | 83.48% (+9.85 points) | 84.63% (-0.45 points) |

Fine-tuning passed the acceptance test (real accuracy improved, synthetic regression well under the 3-point tolerance) — it **was accepted as the new production model**. Unlike the earlier 2-log run (295 held-out windows, flat result), the held-out set is now large enough (5062 windows) for the improvement to be a statistically meaningful result rather than noise, and `EarlyStopping` this time actually moved off epoch 1, converging around epoch 9-10 of 15.

A previous run of this same fine-tuning step crashed here with a `train_test_split` `ValueError`: the "replay" subsample size is capped at `min(len(synthetic_train), 3 * len(real_train))`, and with 11,802 real windows the `3x` term exceeded the entire synthetic training set for the first time (it never had with only ~979 real windows), so the code asked `train_test_split` for a test split equal in size to the whole array, which isn't a valid split. Fixed in `flight_sequence_classifier.py` by using the whole synthetic training set directly (skipping the split) whenever the cap doesn't bind.

One side effect of the larger, differently-composed fine-tuning set worth calling out honestly: on the two original logs specifically, `real_flight.ulg` improved slightly (61.6% → 63.4%, see table below) but `real_flight_2.ulg` **regressed** (69.9% → 59.6%). This isn't fine-tuning breaking — it's the fine-tuning set now being dominated by the four new public logs (11,802 real windows total, `real_flight_2.ulg` contributing only 570 of those), so the model's real-data adaptation is shaped much more by them than by these two original flights. The aggregate real held-out number (73.63% → 83.48%, over a held-out set 17x larger than before) is the trustworthy summary metric here; individual per-log swings are expected when the relative weight of each source flight in the fine-tuning mix changes.

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

Measured with the current model (LSTM, fine-tuned on real data — see [Real-data fine-tuning](#real-data-fine-tuning) above):

| Log | Ground-truth accuracy | Coverage |
|---|---|---|
| `sample.ulg` | n/a (never leaves `MANUAL`) | 0% |
| `real_flight.ulg` | 63.4% | 5.4% (164 windows) |
| `real_flight_2.ulg` | 59.6% | 36.1% (815 windows) |
| `real_flight_3_vtol.ulg` | 0.0% | 18.6% (3662 windows) |
| `real_flight_4_poshold.ulg` | n/a (all `POSCTL`) | 0% |
| `real_flight_5_stab.ulg` | n/a (all `STAB`) | 0% |
| `real_flight_6_takeoff_land.ulg` | 82.6% | 100.0% (3732 windows) |
| `real_flight_7_takeoff_land.ulg` | 93.8% | 100.0% (2284 windows) |
| `real_flight_8_hover_rtl.ulg` | 95.9% | 75.5% (3846 windows) |
| `real_flight_9_hover_land.ulg` | 95.0% | 96.1% (6023 windows) |

A next-mode (forecast) accuracy table is in [Next-mode forecasting](#next-mode-forecasting) above (including a note on why `real_flight_7_takeoff_land.ulg` scores 0% there specifically — a label-quality artifact of that log, not a modeling failure).

`real_flight_3_vtol.ulg`'s 0% is a genuine, verified domain-gap result, not a measurement artifact: during its `AUTO_LOITER`/`AUTO_RTL` segments the aircraft is actually circling in fixed-wing mode at ~16-18 m/s horizontal speed with 27-33° of bank — nothing like the near-stationary, near-level multirotor `hover`/`rtl` the model was trained on. It mostly predicts `anomaly` there instead, which is arguably the "least wrong" label available to it, since the closest true label (`hover`/`rtl`) isn't in its vocabulary for that flight regime.

The four public logs (`real_flight_6` through `real_flight_9`) score much higher (82.6-95.9%) than the two original ones — expected, since they're exactly the flight profiles (clean multirotor takeoff/hover/land/rtl cycles) the fine-tuning set is now dominated by. See [Real-data fine-tuning](#real-data-fine-tuning) above for why `real_flight.ulg`/`real_flight_2.ulg` individually moved up and down rather than both improving — the aggregate real held-out number there is the more meaningful summary than any single log's accuracy.

`data/sample.ulg` is a real PX4 log (from the [pyulog](https://github.com/PX4/pyulog) test suite) used as the example input. Note: this particular log is a stationary bench/attitude test, not an actual flight (near-zero velocity, but large roll swings from the vehicle being manually rotated) — depending on the trained model, it may not predict `hover` throughout, which is a real example of the sim-to-real gap: the synthetic training data never paired large attitude changes with near-zero velocity, since normal flight doesn't produce that combination. Its `nav_state` never leaves `MANUAL`, so it has no ground-truth coverage at all.

Nine genuine flights are also included:

- `data/real_flight.ulg` — a ~10-minute multirotor flight, low horizontal speed throughout (0-3.5 m/s) and a persistent ~-12 deg pitch trim even at hover. Against the original `MODE_PARAMS` (single fixed speed profile, hover pitch centered on 0), this log broke the model in two different, instructive ways: first it predicted `transition` almost everywhere (no travel mode's speed range was ever close to this slow), and after the speed-scale fix alone, it flipped to predicting `descend` almost everywhere (its constant negative pitch trim looked exactly like the `descend` class' pitch mean once slow-mode speeds started overlapping hover). Only after also randomizing pitch trim per flight did the predicted timeline start tracking the actual altitude changes (e.g. the sustained descent in the last ~60s of the flight is correctly predicted as `descend`). Ground truth covers its `AUTO_LOITER`/`AUTO_TAKEOFF`/`AUTO_LAND` segments.
- `data/real_flight_2.ulg` — a shorter (~3.5 min), faster, near-zero-trim multirotor flight (horizontal speed up to 14 m/s). It validates the other end of the widened range: the two fast horizontal bursts early in the flight are correctly picked up as `cruise`/`rtl` instead of being invisible to the model.
- `data/real_flight_3_vtol.ulg` — a ~33-minute VTOL (fixed-wing-capable) test flight, flown mostly in manual `STAB`/`ACRO` modes with only brief `AUTO_LOITER`/`AUTO_RTL`/`AUTO_MISSION` segments. This is a different domain-gap example from the other two: the model was only ever trained on a multirotor hover→takeoff→ascend→cruise→rtl→descend→land mission cycle, so a mostly hand-flown VTOL log is expected to have low ground-truth coverage and is included as an honest stress test, not a validated success case.
- `data/real_flight_4_poshold.ulg` — a ~2.4-minute multirotor flight flown entirely in `POSCTL` (manual position hold). No ground-truth coverage by design (`POSCTL` isn't in the mapping table above — a pilot can translate freely in position hold, so it has no single equivalent label), but it's a real, continuous manual-flight example for judging the timeline by eye.
- `data/real_flight_5_stab.ulg` — a ~1-minute multirotor flight flown entirely in `STAB` (manual stabilized). Same story as above: no ground-truth coverage, included as another real hand-flown segment.
- `data/real_flight_6_takeoff_land.ulg` through `data/real_flight_9_hover_land.ulg` — four public quadrotor logs pulled from [PX4's Flight Review database](https://review.px4.io) (`review.px4.io/dbinfo` + the CDN download URL each entry provides), chosen from ~1300 candidates filtered for a "good"/"great" community rating, zero logged errors, and strong `hover`/`takeoff`/`land`/`rtl` ground-truth coverage (76-100% of each flight). Added specifically to grow the real-data fine-tuning set (see below) beyond just 2 source flights, without needing new hardware. Different vehicles/airframes than `real_flight.ulg`/`real_flight_2.ulg`, which is the point — it reduces how much the fine-tuned model is shaped by any one aircraft's quirks.

Requires `src/flight_sequence_classifier.py` to have been run first (to produce the saved model files).

Run:

```bash
python src/flight_sequence_classifier.py
python src/real_log_inference.py data/real_flight.ulg
```

## app.py

Streamlit UI around the same inference pipeline as `real_log_inference.py`. Upload a `.ulg` file, or pick one of the ten bundled logs (sample bench test, and the nine real flights described above) from the dropdown. Shows flight duration/sample count/mean confidence, a ground-truth accuracy metric when the log has covered `nav_state` segments, the predicted mode distribution, the sensor + mode timeline, and a table of predicted segments with a CSV download button. Also shows a "Next-mode forecast" section (predicted mode ~10 steps ahead, forecast confidence, horizon-shifted ground-truth accuracy when available, and its own segment table/CSV download) when the forecaster artifacts exist.

Requires `src/flight_sequence_classifier.py` to have been run first (to produce the saved model files).

Run:

```bash
streamlit run src/app.py
```

## live_inference.py

A streaming prototype for autonomous decision support: instead of loading a whole `.ulg` file at once, it feeds raw feature rows in one at a time (from a replayed log, or a live MAVLink connection), maintains a rolling window buffer via `LiveWindowBuffer` (reuses `flight_mode_inference.build_windows` directly, so the streaming and offline preprocessing paths can't drift apart), and scores both the current-mode and next-mode models as each new window completes. It only ever prints an advisory line — it never sends anything back to the vehicle; see [Future work](#future-work) for why closing that loop isn't a good idea with this model as-is.

Two sources are implemented:

- **`replay_log_source`** — replays a `.ulg` file's real timestamps (or as fast as possible with `--speed 0`). This is the one actually exercised: `python src/live_inference.py --mode replay --log data/sample.ulg --speed 0 --min-interval 1.0` runs the full pipeline end-to-end against a real log today.
- **`mavlink_source`** — connects to a live MAVLink stream (SITL or a telemetry radio) via `pymavlink`, reading `ATTITUDE` and `LOCAL_POSITION_NED` messages. Written against the documented message fields but **not exercised in this environment** — no PX4 SITL toolchain or live vehicle available here. Needs `pip install pymavlink` and a running SITL (`udp:127.0.0.1:14540` by default) or radio link before it's been proven, not just written.

`--min-interval` (default 1.0s of flight-time) throttles how often windows actually get scored — MAVLink attitude/position messages can arrive tens of times a second, and each `model.predict()` call costs tens of milliseconds regardless of batch size, so scoring every single incoming sample would fall behind in real time for no benefit (mode changes don't need sub-second resolution).

Run:

```bash
python src/live_inference.py --mode replay --log data/sample.ulg --speed 0
python src/live_inference.py --mode mavlink --connection udp:127.0.0.1:14540  # untested here
```

## cpp/

A dependency-free C++ port of the inference path, for the latency-sensitive half of the "otonom karar desteği" (autonomous decision support) idea from `live_inference.py`: Python's `model.predict()` costs tens of milliseconds per call regardless of batch size (see `live_inference.py`'s `--min-interval` throttle, added specifically to work around this), which is fine for a demo but not for a tight control loop. No TFLite/ONNXRuntime here — this machine only has MinGW g++, and those runtimes' prebuilt Windows binaries target MSVC's ABI, which MinGW isn't binary-compatible with. The model is tiny (~18k params, single-layer LSTM(32) + Dense(16) + Dense(9)), so a direct hand-written forward pass is both simpler to get building at all and much faster per call than a general-purpose runtime would be for something this small.

- **`export_weights.py`** — dumps the trained models' weights + `StandardScaler` mean/scale into `weights_current_mode.h` / `weights_next_mode.h` (plain `constexpr float` arrays). Regenerate after every retrain: `python cpp/export_weights.py`.
- **`lstm_model.hpp`** — the forward pass itself: standardize → single-layer LSTM (Keras' gate order: input, forget, cell, output) → Dense+ReLU → Dense+softmax. Same implementation serves both models via a `Weights` struct of pointers.
- **`verify_parity.py`** + **`parity_check.cpp`** — the correctness check that actually matters: dumps real windows and the genuine Keras models' output probabilities, then confirms the C++ forward pass reproduces them. Checked on 200 real windows (100 current-mode + 100 next-mode, from `real_flight_2.ulg`): **0 label mismatches, max probability difference 0.000002** — indistinguishable from float32 rounding noise, not an approximation.
- **`export_replay_csv.py`** + **`main_replay.cpp`** — the C++ equivalent of `live_inference.py --mode replay`: streams a log's raw rows from a CSV, buffers a rolling window, scores both models, prints the same advisory-only lines. Measured (`real_flight_2.ulg`, 2255 windows scored, no throttling): full run in 16ms of CPU time for both models combined — roughly **3-4 microseconds per inference**, against Python/Keras' tens-of-milliseconds-per-call overhead measured earlier. This is the actual payoff of porting the inference path: real headroom for a genuine control-loop rate, not just a marginal speedup.
- No MAVLink ingestion in C++ (unlike the Python prototype, which at least has an untested `mavlink_source`) — vendoring the MAVLink C headers and standing up a SITL connection just to leave it untested wasn't worth it here. Wiring a real MAVLink source in later is just one more row-producer with the same interface as `export_replay_csv.py`'s output, once there's an actual connection to test it against.

`weights_current_mode.h`/`weights_next_mode.h` are generated, committed output — they can silently drift from the actual `.keras` models if someone retrains and forgets to rerun `export_weights.py`. The `cpp-parity` job in `.github/workflows/tests.yml` catches this on every push/PR: it regenerates the headers and fails the build (`git diff --exit-code`) if the committed version doesn't match, then separately builds the C++ engine and reruns the real parity check against the live models - so both "forgot to re-export" and "the forward pass itself is wrong" are covered, not just one or the other.

Build and run (MinGW g++, no other dependencies):

```bash
python cpp/export_weights.py
g++ -std=c++17 -O2 -o cpp/parity_check.exe cpp/parity_check.cpp && python cpp/verify_parity.py && ./cpp/parity_check.exe
g++ -std=c++17 -O2 -o cpp/main_replay.exe cpp/main_replay.cpp
python cpp/export_replay_csv.py data/real_flight_2.ulg && ./cpp/main_replay.exe cpp/replay_data.csv 1.0
```

## Tests

`tests/` has unit tests for the pure/deterministic pieces of the pipeline (domain randomization, quaternion conversion, nav_state ground-truth mapping, windowing incl. the `horizon` shift used by next-mode forecasting, `evaluate_predictions` incl. its horizon-shifted ground-truth alignment, `temporal_split`, prefixed artifact loading, segment summarization, `live_inference.py`'s streaming window buffer and advisory-message logic) — no `.ulg` files or trained model required (artifact loading is tested with `monkeypatch`, not real files). Runs on push/PR via `.github/workflows/tests.yml`.

```bash
pytest
```

## Future work

~~The real-data fine-tuning set is currently small (~979 ground-truth-labeled windows from 2 logs, covering only 4 of 9 modes).~~ **Done**: 4 more public logs from PX4's Flight Review database were added (see `real_flight_6`-`real_flight_9` above), growing real ground-truth coverage roughly 6x. This still only covers `hover`/`takeoff`/`land`/`rtl` — `AUTO_MISSION`/`POSCTL`/`ACRO`/`STAB` nav_states remain deliberately unmapped (see the ground-truth table above), since they don't correspond to a single one of our 9 labels. Actually adding coverage for the other 5 classes (`ascend`/`cruise`/`descend`/`transition`/`anomaly`) would need a different approach than "find more public logs" — e.g. inferring sub-legs of `AUTO_MISSION` from waypoint/altitude-setpoint data, which is a real design problem, not just a data-collection one. PX4 SITL (Gazebo/jMAVSim) remains a hardware-free option for generating more real-PX4-state-machine logs locally in the meantime.

`live_inference.py` and `cpp/` (above) are deliberately advisory-only, not a controller wired into the autopilot. Closing that loop would need: real ground-truth coverage for the `anomaly` class (currently 100% synthetic — the one class where being wrong matters most), a measured/bounded inference latency budget on the *actual target hardware* (the ~3-4 microsecond/inference number in `cpp/` is from this dev machine, not a companion computer like a Raspberry Pi or Jetson — still unmeasured there, and MAVLink message parsing/buffering overhead isn't included either), and a case for why a probabilistic classifier should override PX4's own deterministic mode state machine at all, when it already handles mode transitions reliably. None of that exists yet, so the safe scope for now is exactly what's implemented: printing forecasts a human or a downstream advisory system can act on.
