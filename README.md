# UAV Fault Precursor Detector

A loss-of-control precursor detector for ArduPilot drones, built to run in real time. It reads four sensor values (vertical speed, horizontal speed, roll, pitch) and outputs the current fault state plus a ~1-2 second forecast of the next one, across five classes: `normal`, `motor_out`, `gps_glitch`, `wind_gust_upset`, `sensor_freeze`. The model is an LSTM trained on synthetic flights with physically-motivated fault profiles. The inference path is also ported to dependency-free C++ for real-time/low-latency use.

This project used to classify PX4 flight mode (hover/takeoff/cruise/...). That framing was dropped: an autopilot already reports its own current mode natively, so predicting it from sensor data is redundant. What an autopilot does *not* do is warn you a fault is coming before it fully manifests - that is this project's actual job now. The target autopilot also switched from PX4 to ArduPilot; the project's earlier PX4-based real flight logs and PX4-specific code no longer exist in this repo (recoverable from git history if ever needed).

Read in this order: [fault_sequence_classifier.py](#fault_sequence_classifierpy) (train the model) → [real_log_inference.py](#real_log_inferencepy) (check it against a real log) → [live_inference.py](#live_inferencepy) and [cpp/](#cpp) (run it in real time) → [app.py](#apppy) (a Streamlit UI for poking at all of the above). [fault_classifier_baseline.py](#baseline-fault_classifier_baselinepy-superseded) is an earlier, superseded approach kept for comparison.

The live demo has not been redeployed for this pivot yet - if you find a link to one elsewhere, assume it still reflects the old PX4 mode-classifier and ignore it.

## Setup

```bash
pip install -r requirements.txt
```

## flight_data.py

Shared constants for synthetic flight generation: `FEATURES`, `MODE_CYCLE`, and `MODE_PARAMS` (per-mode sensor mean/std, used only to generate a realistic *background* flight for fault segments to be injected into - mode identity itself is no longer a supervised label anywhere in this project).

`randomize_mode_params()` applies two per-flight randomizations discovered necessary against real PX4 logs in the project's earlier phase and kept because they still produce more realistic background variety than a single fixed profile: `SPEED_SCALE_RANGE` (a random per-flight speed multiplier on the "travel" modes) and `PITCH_TRIM_RANGE` (a random constant pitch offset, simulating airframe rigging/CG trim).

## fault_injection.py

Four physically-motivated fault-precursor segment generators, replacing an earlier pure-Gaussian-noise `anomaly` class that had no temporal structure and no real ground-truth coverage:

- `sample_motor_out_segment` - asymmetric roll/pitch divergence plus a vertical-speed drop, with an exponential onset ramp rather than a step jump. Labeling the ramp itself as `motor_out` from onset (not just the fully-diverged plateau) is what makes forecasting it meaningful.
- `sample_gps_glitch_segment` - a step discontinuity in horizontal_speed with a decaying-noise recovery as the EKF would re-converge; roll/pitch stay near background.
- `sample_wind_gust_upset_segment` - a transient roll + horizontal_speed spike that only partially decays back toward baseline by segment end. The partial recovery is what separates it from `motor_out`, which doesn't recover.
- `sample_sensor_freeze_segment` - flatlines one randomly chosen feature at a time at its last value (zero variance and zero delta on that feature only), matching a single real sensor failing rather than all four features going flat at once.

Each generator is injected against whatever mode/speed-scale/pitch-trim background context `fault_sequence_classifier.generate_flight` happens to be in at the insertion point, not as a disconnected blob.

## fault_sequence_classifier.py

Trains the model. Time-series classification over 15-step sliding windows of sensor readings.

- Dataset: 200 synthetic flights, each cycling through the background mode sequence with 0-2 fault segments independently drawn and inserted per flight.
- Features: 4 raw sensor readings plus their timestep deltas, 8 total.
- Compares four architectures (LSTM, GRU, Conv1D, BiLSTM) the same way the project always has: 64-unit recurrent layers with `recurrent_dropout=0.1` (Conv1D gets a plain `Dropout(0.1)` between its two causal conv layers instead), all sharing a `Dropout(0.2)` + `Dense(32, relu)` head.
- Split by flight, not by window, so overlapping windows can't leak across train/test.
- Class-weighted loss for the rare fault classes; `EarlyStopping` on validation accuracy for every CV fold and the final run.
- 5-fold flight-level CV runs before the final split.

Dataset composition (46,052 total timesteps):

| Class | Timesteps |
|---|---|
| `normal` | 40,725 |
| `wind_gust_upset` | 1,444 |
| `motor_out` | 1,388 |
| `sensor_freeze` | 1,312 |
| `gps_glitch` | 1,183 |

Results:

| Architecture | Test accuracy | 5-fold CV |
|---|---|---|
| BiLSTM | 91.05% | 86.61% ± 5.15% |
| Conv1D | 86.17% | 65.31% ± 7.76% |
| GRU | 83.38% | 77.26% ± 4.55% |
| LSTM | 78.07% | 83.25% ± 7.22% |

`sensor_freeze` is the hardest class everywhere - on the pinned LSTM it scores 0.07 precision / 0.52 recall: the model flags it too eagerly and is usually wrong when it does. A single flatlined feature against an otherwise-quiet background (e.g. hover, which is already low-variance) is genuinely hard to tell apart from ordinary quiet flight with this feature set alone.

BiLSTM wins by a wide margin this run, but production is **pinned to LSTM regardless**: `cpp/lstm_model.hpp` only implements a single-layer unidirectional LSTM forward pass, and shipping a BiLSTM would break C++ parity. This costs real accuracy (78.07% vs 91.05%) - worth stating plainly rather than glossing over, since it's a real and current tradeoff, not a historical one.

This comparison is run on the nowcast task only, and the winner is reused for the forecaster without separately checking whether it still wins there. That matters here specifically: a bidirectional architecture's backward pass starts at the window's *last* timestep, which for nowcasting is the exact point being labeled - it gets that timestep's raw signal before any recurrent decay, an advantage a unidirectional LSTM only accumulates after 14 steps of forgetting. For the forecaster, the labeled point is 10 steps past the window's end and isn't in the window at all, in either direction, so this advantage may not carry over. BiLSTM's win here is real but may be partly an artifact of the easier task, not evidence it's the better architecture for the harder one this project actually cares about - not re-measured for the forecaster, since the C++ constraint pins production to LSTM regardless of the answer.

The pinned LSTM has ~20,933 parameters (single-layer LSTM(64) + Dense(32) + Dense(5)). Running this script saves `models/fault_model.keras`, its scaler, and metadata, plus a second forecasting model.

Run:

```bash
python src/fault_sequence_classifier.py
```

### Fault forecaster

The same script also trains a second model that forecasts the fault state 10 steps ahead (`PREDICTION_HORIZON = 10`, roughly 1-2 seconds depending on sample rate) from the same 15-step window - only the label shifts forward.

- Saved as `models/fault_next_model.keras` / `..._scaler.joblib` / `..._meta.joblib`.
- Test accuracy: 55.52%, vs 78.07% for nowcasting on the same split. Forecasting ahead is a harder problem across every class: `motor_out` precision drops to 0.15, `sensor_freeze` to 0.03. `gps_glitch` and `wind_gust_upset` hold up comparatively better (recall 0.61 and 0.48).
- This is the model doing the actual job this project exists for - naming a fault before it fully manifests - and its current accuracy should be read as a first, synthetic-only baseline, not a finished result.

### SITL fine-tuning

The CV-winning-for-parity (LSTM) model gets a fine-tuning pass on ground-truth-labeled ArduPilot SITL windows, gated so real data can only improve the production model, never quietly regress it - same acceptance-test structure the project has always used: fine-tune on real + a stratified synthetic replay sample, accept only if real held-out accuracy doesn't drop and synthetic accuracy doesn't regress by more than 3 points.

**No SITL logs exist yet.** `SITL_FINETUNE_LOGS` in this file lists the paths `scripts/sitl_generate_fault_logs.py` (see below) is meant to produce; until you generate them, both the nowcaster and forecaster print "not found, skipping" for every entry and stay on the synthetic-only model. This is not a bug - it is the honest current state. No ArduPilot equivalent of PX4's public Flight Review log database exists to mine instead, so SITL generation is the only realistic real-data source for v1.

## inference_common.py / ardupilot_log.py

The evaluation/inference pipeline is split into an autopilot-agnostic half and an ArduPilot-specific half.

`inference_common.py` - `build_windows`, `temporal_split`, `evaluate_predictions`, `load_artifacts`, `predict`, `summarize_segments`, `build_timeline_figure`, `attach_ground_truth`, `extract_labeled_real_windows`. None of these read a specific log format or fault signal; they operate on whatever strings are in a `ground_truth_mode` column, and `extract_labeled_real_windows` takes its log-loading and ground-truth functions as parameters.

`ardupilot_log.py` - the ArduPilot-native half:

- `load_flight_log` reads a `.bin` dataflash log via `pymavlink.mavutil.mavlink_connection` (which auto-detects the dataflash format from the file itself). `roll_angle`/`pitch_angle` come directly from `ATT.Roll`/`ATT.Pitch` in degrees - no quaternion math needed, simpler than PX4's `vehicle_attitude`. `vertical_speed` comes from `CTUN.CRt`; `horizontal_speed` from `GPS.Spd`. **Not yet verified against a real logged flight**: `CTUN.CRt`'s sign convention, and whether an EKF-fused velocity field would track the live MAVLink path's `LOCAL_POSITION_NED.vx/vy` better than raw GPS ground speed - both need checking once real SITL logs exist.
- `load_scripted_fault_ground_truth` reads the `<log>.fault_windows.json` sidecar that `scripts/sitl_generate_fault_logs.py` writes alongside each generated log, and labels every row `normal` or the named fault by elapsed flight time. This is the primary, exact ground truth source, and the only one for `wind_gust_upset`/`sensor_freeze` (ArduPilot has no native failsafe signal for either).
- `load_native_fault_ground_truth` reads ArduPilot's own `ERR` messages, gated on the first `EV` armed event so boot-time transients aren't mislabeled as faults. Only `ERR.Subsys` code 25 (`THRUST_LOSS_CHECK`, ArduCopter's own motor/propulsion-failure detector) is mapped to `motor_out` right now - it's the one confirmed-by-name code; the rest of ArduPilot's subsystem enum (which would give `gps_glitch` a native source too) needs confirming against a real logged `ERR` message before being added.
- `load_fault_ground_truth` tries the sidecar first, falls back to the native `ERR` mapping.

## real_log_inference.py

Runs the saved model on a real `.bin` log instead of synthetic data: parses it via `ardupilot_log`, builds the same windows, predicts a fault class for each, and reports ground-truth accuracy where a sidecar or `ERR` mapping is available.

Requires `fault_sequence_classifier.py` to have been run first, and a real `.bin` log to point at (none are bundled yet - see SITL fine-tuning above).

Run:

```bash
python src/fault_sequence_classifier.py
python src/real_log_inference.py data/sitl_motor_out_1.bin
```

## live_inference.py

A streaming version of the same pipeline: feeds feature rows in one at a time (replayed log or live MAVLink) instead of loading a whole file, maintains a rolling window via `LiveWindowBuffer`, and scores both models as each window completes. Advisory only - it never writes back to the vehicle.

Two sources:

- `replay_log_source` - replays a `.bin` file's real timestamps.
- `mavlink_source` - connects to a live MAVLink stream via `pymavlink`, reading `ATTITUDE`/`LOCAL_POSITION_NED`. These are common MAVLink messages, identical on ArduPilot and PX4, so this path needed no logic changes for the autopilot switch. It was, however, missing a `request_data_stream_send` call - without it, neither ArduPilot nor PX4 pushes these messages to a fresh client at all. Found and fixed by actually connecting to a live SITL instance (see `scripts/sitl_live_monitor.py`), not by inspection.

`advisory()`'s urgency check changed with the new label set: any forecast that isn't `normal` and clears a confidence threshold is flagged urgent, and a forecast recovering back to `normal` is informational rather than silent.

Run:

```bash
python src/live_inference.py --mode replay --log data/sitl_motor_out_1.bin --speed 0
python src/live_inference.py --mode mavlink --connection tcp:127.0.0.1:5760  # ArduPilot SITL's default serial0; use udp:127.0.0.1:14550 if launched via MAVProxy
```

## cpp/

A dependency-free C++ port of the inference path, unchanged in substance by either pivot - `cpp/lstm_model.hpp` is fully generic over class names and count, so repurposing it from 9 mode classes to 5 fault classes was a matter of retraining and re-exporting, not touching the forward pass.

- `export_weights.py` - dumps trained weights + scaler into `weights_current_fault.h` / `weights_next_fault.h`. Regenerate after every retrain.
- `lstm_model.hpp` - the forward pass: standardize → single-layer LSTM (Keras gate order) → Dense+ReLU → Dense+softmax.
- `verify_parity.py` + `parity_check.cpp` - confirms the C++ forward pass matches Keras on real windows. **Verified against a real ArduPilot `.bin` log** (a pre-existing bench-test SITL flight, not one of this project's own fault logs): 200 windows checked, 0 label mismatches, max probability difference 0.000001.
- `benchmark.cpp` - isolated latency measurement, needs no real log. Freshly measured on this dev machine after the pivot: **~306 microseconds for both models combined, ~153 microseconds per single-model inference** - comfortably inside a 100Hz (10ms) control-loop budget. Not directly comparable to the project's earlier ~460us figure from before the pivot; that was a different model (9 classes) on different hardware.
- `export_replay_csv.py` + `main_replay.cpp` - the C++ equivalent of `live_inference.py --mode replay`.

`weights_current_fault.h`/`weights_next_fault.h` are generated output that can drift from the actual `.keras` models if someone retrains and forgets to re-export. The `cpp-parity` job in `.github/workflows/tests.yml` catches this on every push by regenerating the headers and failing the build if they don't match, then rebuilding and rerunning the parity check against the live models - that job will fail until a real log exists for `verify_parity.py` to run against.

Build and run (MinGW/GCC, no other dependencies):

```bash
python cpp/export_weights.py
g++ -std=c++17 -O2 -o cpp/parity_check.exe cpp/parity_check.cpp && python cpp/verify_parity.py && ./cpp/parity_check.exe
g++ -std=c++17 -O2 -o cpp/benchmark.exe cpp/benchmark.cpp && ./cpp/benchmark.exe
g++ -std=c++17 -O2 -o cpp/main_replay.exe cpp/main_replay.cpp
python cpp/export_replay_csv.py data/sitl_motor_out_1.bin && ./cpp/main_replay.exe cpp/replay_data.csv 1.0
```

## app.py

A Streamlit UI for inspecting the pipeline without a MAVLink connection or a build step. Upload a `.bin` file or pick a bundled SITL log (none bundled yet - see SITL fine-tuning above). Shows duration/sample count/mean confidence, ground-truth accuracy where available, fault distribution, a sensor+fault timeline, a segment table with CSV download, and a fault forecast section.

Requires `fault_sequence_classifier.py` to have been run first.

Run:

```bash
streamlit run src/app.py
```

## Baseline: fault_classifier_baseline.py (superseded)

The independent-reading (no window, no memory of the rest of the flight) approach, retargeted at the new fault taxonomy after the pivot - kept because the argument for why this doesn't work is, if anything, stronger for fault detection than it was for mode classification: a single instant's roll/pitch can't distinguish "a wind gust in progress" from "just banking," any more than it could distinguish a mode from a trim offset.

- Dataset: synthetic, drawing from the same `fault_injection` generators used by the sequence model, collapsed to independent rows. Each fault class is drawn from 15 independent realizations (separate mode/speed-scale/pitch-trim/noise-scale/frozen-value draws) rather than one, and the train/test split is grouped by realization (`GroupShuffleSplit`) - an earlier ungrouped version let `sensor_freeze` rows from the same single realization leak its one frozen value across both the train and test split, inflating its score to a meaningless 1.00 precision/recall by memorizing a constant instead of learning anything.
- Algorithm: Random Forest.

With that leakage fixed, the result actually makes the "why we moved to a sequence model" case concretely: **84.08% overall accuracy, but `sensor_freeze` precision/recall are both 0.00** - a single reading genuinely cannot tell "this value happens to be constant across recent samples" from "this value could belong to any class," since that's a property of a *sequence*, not an instant. `wind_gust_upset` (0.85/0.56) and `gps_glitch` (0.40/0.70) fare better since their shapes (a spike, a step) partly show up in on-the-spot feature values, but still lag the sequence model by a wide margin.

Run:

```bash
python src/fault_classifier_baseline.py
```

## scripts/sitl_generate_fault_logs.py

Connects to a running ArduPilot SITL instance via MAVLink, arms and takes off, holds a normal background pattern, injects one of the four fault types via SITL simulation parameters, holds it, clears it, lands, and copies the resulting dataflash log into `data/` alongside a `.fault_windows.json` sidecar recording exactly when the fault was active.

| Fault | SITL parameters |
|---|---|
| `motor_out` | `SIM_ENGINE_FAIL` (motor bitmask) + `SIM_ENGINE_MUL` (thrust multiplier, stepped to 0) |
| `gps_glitch` | `SIM_GPS1_VERR_X`/`SIM_GPS1_VERR_Y` (direct GPS velocity error injection) |
| `wind_gust_upset` | `SIM_WIND_SPD`/`SIM_WIND_DIR`/`SIM_WIND_TC` |
| `sensor_freeze` | `SIM_BARO_FREEZE` |

Not yet run to completion end to end: a real ArduPilot SITL instance was available in this environment and the MAVLink connection, arming, and mode-switching were all confirmed to work individually, but the scripted arm-then-takeoff sequence didn't reliably leave the vehicle armed long enough to fly a real fault profile - see `sitl_live_monitor.py` below and Future Work. Prerequisite for running this elsewhere: a working `ArduPilot/ardupilot` checkout with `sim_vehicle.py` set up per ArduPilot's own SITL docs.

Run (once SITL is running separately):

```bash
python scripts/sitl_generate_fault_logs.py --fault motor_out --count 10 --sitl-log-dir ~/ardupilot/logs
```

## scripts/sitl_live_monitor.py

A live dashboard for watching the model work against a real MAVLink connection in real time: flies the same background/inject/hold/clear/recover sequence as `sitl_generate_fault_logs.py`, scores every completed window with both models as telemetry arrives, and serves a small auto-refreshing web page (`http://localhost:8765` by default) showing the current prediction, the forecast, the known injected fault state, and a running nowcast accuracy.

This was actually run against a live local SITL instance, and it's how several real bugs got found and fixed, not just theorized about:

- `live_inference.py`'s `mavlink_source` never requested MAVLink data streams after connecting - ArduPilot (and PX4) don't push `ATTITUDE`/`LOCAL_POSITION_NED` to a fresh client without an explicit `request_data_stream_send`. Without this fix, the live path would have received nothing on a real connection. Fixed in both `mavlink_source` and this script.
- The first version of this script ran MAVLink message reading and model inference (`model.predict()`, documented elsewhere as tens of milliseconds per call) on the same thread. This starved message processing during inference. Fixed by splitting reading and scoring into separate threads connected by a queue - reading must never block on inference.
- The arm command was sent exactly once and never retried - if ArduCopter's prearm checks reject it (e.g. IMUs not yet settled after a fresh SITL boot), it just stays rejected forever. Fixed by resending every few seconds until `HEARTBEAT` confirms armed, with `COMMAND_ACK`/`STATUSTEXT` (`STATE["last_arm_ack"]`/`STATE["last_statustext"]`) surfaced so a rejection reason is visible instead of a silent timeout. The same fix (and the same diagnostic pattern, `STATE["last_takeoff_ack"]`) is applied to `MAV_CMD_NAV_TAKEOFF` - which surfaced the still-open issue below rather than fixing it, since a takeoff rejection needs a different response (wait for the actual precondition) than an arm rejection does (just wait and retry).

What actually got validated live: the MAVLink connection, stream request, windowing, and dual-model scoring all work end to end against a real SITL vehicle, arming itself succeeds with the retry fix, and **the model consistently predicted `sensor_freeze` at ~80%+ confidence while the vehicle sat still on the ground** - a live confirmation of the weakness already measured on synthetic data (a quiet, low-variance moment looks like a frozen sensor). What still isn't validated: whether the model correctly reacts to an actual in-flight fault - `MAV_CMD_NAV_TAKEOFF` itself is being rejected (see Future Work), so the vehicle never actually leaves the ground yet.

Run (once SITL is running separately):

```bash
python scripts/sitl_live_monitor.py --connection tcp:127.0.0.1:5760
```

## Tests

Unit tests for the deterministic pieces: domain randomization, fault-segment generator shapes/trends, the ArduPilot ground-truth extraction functions (against synthetic fixtures, no real `.bin` file needed), windowing (including the horizon shift), `evaluate_predictions`, `temporal_split`, artifact loading, segment summarization, and `live_inference.py`'s buffer/advisory logic. Runs on push/PR.

```bash
pytest
```

## Future work

- **The scripted arm/takeoff sequence doesn't reliably get the vehicle off the ground.** This was investigated deeply, not just guessed at, and one real bug is fixed, but the core issue remains open:
  1. *Fixed*: the arm command was being rejected outright (`STATUSTEXT`: "Arm: Accels inconsistent" - ArduCopter's prearm check rejects arming until SITL's simulated IMUs settle after a fresh boot) and was only ever sent once, so it never got retried once the check cleared a few seconds later. Fixed in both scripts by resending the arm command every few seconds until `HEARTBEAT` confirms armed.
  2. *Root cause found via direct source instrumentation, still unresolved*: once armed, `MAV_CMD_NAV_TAKEOFF` comes back `COMMAND_ACK` result 4 (rejected). Several plausible preconditions were checked live and ruled out one by one: EKF position flags (`EKF_STATUS_REPORT.flags`) were already ok, `landed_state` (`EXTENDED_SYS_STATE`) correctly reported ON_GROUND, and a `HOME_POSITION` message had already been received. A local ArduCopter source checkout (not part of this repo) was temporarily instrumented with `GCS_SEND_TEXT` debug lines at every `return false` in `Mode::do_user_takeoff_U_m`/`ModeGuided::do_user_takeoff_start_m` (`ArduCopter/takeoff.cpp`, `ArduCopter/mode_guided.cpp`) and rebuilt (reverted afterward, not committed anywhere) - this gave a definitive answer instead of another guess: the rejection reason varies between `has_user_takeoff false` (impossible in GUIDED mode by that mode's own code, meaning the vehicle wasn't actually in GUIDED at that instant) and `not armed` - **the vehicle's mode/armed state is flapping during the sequence, not stuck on one static precondition.** A hypothesis that our script never sends its own GCS `HEARTBEAT` (unlike MAVProxy, which does continuously, and which sends an identical `MAV_CMD_NAV_TAKEOFF` packet - confirmed by reading MAVProxy's own `cmd_takeoff` source) was tested by adding a 1Hz heartbeat-send loop to both scripts - this did not resolve it either. The actual trigger for the flapping is still unknown. Next step if picked up again: attach `gdb` to the SITL binary at the exact rejection point rather than inferring from MAVLink telemetry - `STATUSTEXT` debug lines proved which branch fails but not the live variable state (e.g. `copter.current_loc`, AHRS origin, active failsafes) behind it.
- **SITL fine-tuning has never run** as a result - both models are still 100% synthetic-only in production. The live pipeline itself (connection, streaming, windowing, dual-model scoring) is confirmed working; what's missing is a flight that actually stays armed and airborne through a fault injection.
- **Live-confirmed: `sensor_freeze` over-triggers on genuinely quiet flight**, not just in synthetic test data - a real SITL vehicle sitting still on the ground was classified `sensor_freeze` at ~80%+ confidence essentially the whole time. This is the same weakness the synthetic numbers already showed (0.07 nowcast / 0.03 forecast precision), now also seen on real telemetry. A single flatlined feature against a quiet background is genuinely hard to distinguish from ordinary quiet flight with only 4 sensor values. An attempted fix (a per-flight "quiet normal" domain randomization added to the synthetic training data) was tried and reverted - it reduced the false-positive rate on quiet backgrounds but crashed recall for every class, including `sensor_freeze` itself (0.52 → 0.11 on the pinned LSTM) and the forecaster overall (55.52% → 39.17%), a net regression on every measured number. Still unresolved.
- **The forecaster is a first synthetic-only baseline** (55.52% vs 78.07% nowcasting) - expect it to move once SITL fine-tuning actually runs.
- **ArduPilot's `ERR.Subsys` enum is only partially mapped** - only code 25 (`THRUST_LOSS_CHECK`) is confirmed by name; the codes that would give `gps_glitch` a native (non-sidecar) real-log ground truth source need confirming against an actual logged flight.
- **No public ArduPilot log database exists** to mine for real-world fault examples the way PX4's Flight Review database was used before the pivot - SITL generation is the only realistic v1 real-data path.
- `live_inference.py` and `cpp/` are advisory-only by design, not wired into the autopilot. Closing that loop needs real fault ground truth at scale, a measured latency budget on actual target hardware (a Raspberry Pi/Jetson, not this dev machine), and a case for why a probabilistic classifier should override the autopilot's own failsafe state machine at all.
