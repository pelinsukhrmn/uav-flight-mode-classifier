# UAV Fault Precursor Detector

A loss-of-control precursor detector for ArduPilot drones, built to run in real time. It reads nine signals the autopilot already logs and outputs the current fault state plus a ~1-2 second forecast of the next one, across five classes: `normal`, `motor_out`, `gps_glitch`, `wind_gust_upset`, `sensor_freeze`. The model is an LSTM trained on synthetic flights and fine-tuned on labeled ArduPilot SITL flights. The inference path is also ported to dependency-free C++ for real-time/low-latency use.

The headline result, measured on SITL flights the model never trained on: it names `motor_out` **9.7 seconds before ArduPilot's own `THRUST_LOSS_CHECK` fires**, with zero false `motor_out` alarms across a fault-free control flight. `gps_glitch` scores 0.91 precision / 0.92 recall and `motor_out` 0.83 / 0.93 on held-out flights. `sensor_freeze` scores 0.00 - it is not observable in this setup, and the Future work section explains why rather than hiding it.

This project used to classify PX4 flight mode (hover/takeoff/cruise/...). That framing was dropped: an autopilot already reports its own current mode natively, so predicting it from sensor data is redundant. What an autopilot does *not* do is warn you a fault is coming before it fully manifests - that is this project's actual job now. The target autopilot also switched from PX4 to ArduPilot; the project's earlier PX4-based real flight logs and PX4-specific code no longer exist in this repo (recoverable from git history if ever needed).

Read in this order: [fault_sequence_classifier.py](#fault_sequence_classifierpy) (train the model) → [real_log_inference.py](#real_log_inferencepy) (check it against a real log) → [live_inference.py](#live_inferencepy) and [cpp/](#cpp) (run it in real time) → [app.py](#apppy) (a Streamlit UI for poking at all of the above). [fault_classifier_baseline.py](#baseline-fault_classifier_baselinepy-superseded) is an earlier, superseded approach kept for comparison.

The live demo has not been redeployed for this pivot yet - if you find a link to one elsewhere, assume it still reflects the old PX4 mode-classifier and ignore it.

## Setup

```bash
pip install -r requirements.txt
```

## flight_data.py

Shared constants for synthetic flight generation: `FEATURES`, `MODE_CYCLE`, and `MODE_PARAMS` (per-mode sensor mean/std, used only to generate a realistic *background* flight for fault segments to be injected into - mode identity itself is no longer a supervised label anywhere in this project).

`FEATURES` holds nine signals. Four are the original raw sensor readings (`vertical_speed`, `horizontal_speed`, `roll_angle`, `pitch_angle`); five were added after measuring, on real SITL logs, which additional signals actually separate the fault classes:

| Feature | Source | Measured separation (median, `normal` vs fault) |
|---|---|---|
| `motor_spread` | `RCOU` C1-C4 max minus min | 1.0 vs 655.0 for `motor_out` |
| `ekf_vel_innov` | `XKF4.SV`, the EKF velocity innovation test ratio | 0.000 vs 1.340 for `gps_glitch` |
| `baro_climb_rate` | derivative of raw `BARO.Alt` | frozen baro reads exactly 0.0000 vs 0.0354 healthy |
| `roll_track_err` | `ATT.DesRoll` minus `ATT.Roll` | 0.01° vs 7.39° for `motor_out` |
| `pitch_track_err` | `ATT.DesPitch` minus `ATT.Pitch` | 0.01° vs 9.28° for `motor_out` |

Three candidate signals were measured and **rejected** before any code was written around them: `VIBE` (SITL does not simulate vibration - median 0.0 in every class), `XKF4.SP` and `XKF4.SH` (no separation between any classes), and window-level statistics such as within-window standard deviation (separated only `motor_out` and `gps_glitch`, which the features above already cover).

The two tracking-error features matter for positioning, not just accuracy: `motor_spread` and `ekf_vel_innov` are the same signals ArduPilot's own `THRUST_LOSS_CHECK` and GPS failsafe consume, whereas `ATT.DesRoll - ATT.Roll` is something the autopilot computes for its control loop but never evaluates for fault classification.

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

- Dataset: 200 synthetic flights, each cycling through the background mode sequence with 0-2 fault segments independently drawn and inserted per flight. Each fault segment lasts 30-70 timesteps - long enough that most of it falls outside the forecaster's fixed 10-step "structurally unlearnable" window (see Fault forecaster below), which was the actual lever that improved both models here, not just the forecaster.
- Features: 9 signals plus their timestep deltas, 18 total.
- Compares four architectures (LSTM, GRU, Conv1D, BiLSTM) the same way the project always has: 64-unit recurrent layers with `recurrent_dropout=0.1` (Conv1D gets a plain `Dropout(0.1)` between its two causal conv layers instead), all sharing a `Dropout(0.2)` + `Dense(32, relu)` head.
- Split by flight, not by window, so overlapping windows can't leak across train/test.
- Class-weighted loss for the rare fault classes; `EarlyStopping` on validation accuracy for every CV fold and the final run.
- 5-fold flight-level CV runs before the final split.

Dataset: 49,974 total synthetic timesteps.

Results (synthetic test split):

| Architecture | Test accuracy | 5-fold CV |
|---|---|---|
| BiLSTM | 97.29% | 95.59% ± 1.50% |
| GRU | 96.89% | 95.85% ± 0.92% |
| LSTM | 96.59% | 95.15% ± 1.07% |
| Conv1D | 92.05% | 94.02% ± 1.70% |

These synthetic numbers jumped roughly ten points when the five new features were added, and that is exactly as meaningful as it sounds: the synthetic generator produces those features itself, so the model is being scored on data built from the same assumptions it trains on. **The synthetic test accuracy is a self-consistency check, not evidence the detector works.** The numbers that count are on real SITL flights, in the SITL fine-tuning section below - and there the synthetic-only model scores 7.11%, which is the honest measure of how far synthetic-to-real transfer goes on its own.

`sensor_freeze` remains the hardest class even synthetically (0.40 precision / 0.40 recall on the pinned LSTM) and scores 0.00 on real held-out flights. See Future work: it is a data-generation problem, not a model problem.

BiLSTM wins this run, but production stays pinned to LSTM because `cpp/lstm_model.hpp` only implements a single-layer unidirectional LSTM forward pass. The cost of that choice is currently 0.70 points of synthetic accuracy, and the comparison is re-run every time rather than assumed.

This comparison is run on the nowcast task only, and the winner is reused for the forecaster without separately checking whether it still wins there - see the caveat about bidirectional architectures having a structural nowcast-only advantage in the Fault forecaster section below.

The pinned LSTM has ~20,933 parameters (single-layer LSTM(64) + Dense(32) + Dense(5)). Running this script saves `models/fault_model.keras`, its scaler, and metadata, plus a second forecasting model.

Run:

```bash
python src/fault_sequence_classifier.py
```

### Fault forecaster

The same script also trains a second model that forecasts the fault state 10 steps ahead (`PREDICTION_HORIZON = 10`, roughly 1-2 seconds depending on sample rate) from the same 15-step window - only the label shifts forward.

- Saved as `models/fault_next_model.keras` / `..._scaler.joblib` / `..._meta.joblib`.
- Synthetic test accuracy: **87.48%**, vs 96.59% for LSTM nowcasting on the same split. `motor_out` precision/recall: 0.72/0.79. `wind_gust_upset`: 0.66/0.69. `gps_glitch`: 0.67/0.80. `sensor_freeze`: 0.15/0.22.
- On real SITL held-out flights it reaches 0.452 macro-F1 after fine-tuning (up from 0.265 synthetic-only) and **fails the acceptance gate**, so production keeps the synthetic-only forecaster. The nowcaster passes the same gate. Forecasting a fault before it manifests is measurably harder than naming one already in progress, and this project does not yet have a forecaster worth shipping.
- Fault segments were lengthened from 20-40 to 30-70 timesteps specifically to help this task, and it worked (up from 55.52% before): with `PREDICTION_HORIZON=10` fixed, the earliest 10 timesteps of *every* fault segment are structurally unlearnable for the forecaster - a window whose forecast target lands in those first 10 steps contains zero fault signal of its own (the target hasn't started yet when the window ends), so the model is being asked to predict something not present in its input at all. That's a fixed-size blind spot regardless of segment length, so lengthening segments shrinks its *share* of the labeled data (from up to half of a 20-step segment down to as little as a seventh of a 70-step one) without changing anything about the `normal` background.
- This also explains why nowcasting improved alongside it even though nowcasting has no such blind spot: longer segments mean less segment-boundary dilution per window generally, and more total fault-class training data at a fixed synthetic flight count.
- Because the architecture comparison earlier is run on the nowcast task only, its winner might not actually be the best choice for this harder task - a bidirectional architecture's backward pass starts at the window's *last* timestep, which for nowcasting is the exact point being labeled, but for forecasting the labeled point isn't in the window at all in either direction. BiLSTM's nowcast win (before this change) may have been partly an artifact of that structural advantage rather than genuine sequence-modeling superiority; not re-measured for the forecaster specifically, since the C++ constraint pins production to LSTM regardless of the answer.
- This is the model doing the actual job this project exists for, and it is the part that does not work yet. The nowcaster's 9.7-second lead over ArduPilot's own detector currently comes from naming a fault that has already started, faster than the autopilot's conservative threshold does - not from forecasting one that hasn't.

### SITL fine-tuning

The pinned production (LSTM) model gets a fine-tuning pass on ground-truth-labeled ArduPilot SITL windows, gated so real data can only improve the production model, never quietly regress it.

`SITL_FINETUNE_LOGS` globs `data/sitl_*.bin`, so every generated flight is picked up automatically. The current dataset is 15 flights: three each of `motor_out`, `wind_gust_upset` and `sensor_freeze`, two `gps_glitch`, and four fault-free control flights.

Two parts of this evaluation were wrong until they were fixed, and both had silently inflated the numbers:

- **The held-out split used to be temporal** - the last 30% of each flight. Faults are injected mid-flight, so that tail was 100% `normal` and the majority-class baseline on it was 100.00%: the "real data accuracy" being reported measured nothing but the model's willingness to say `normal`. It now holds out **whole flights**, stratified so each fault class and at least one control flight land in the holdout.
- **The acceptance gate used to compare accuracy**, which on a holdout that is 90.41% `normal` rewards a constant predictor over a real detector. It now compares macro-F1 against the majority-class baseline's macro-F1 (0.190).

Results on the five held-out flights (7,185 windows, never seen in training):

| | Macro-F1 | Accuracy |
|---|---|---|
| Majority-class baseline | 0.190 | 90.41% |
| Synthetic-only model | 0.344 | 7.11% |
| Fine-tuned model (production) | **0.662** | 88.53% |

Per class on those same held-out flights:

| Class | Precision | Recall |
|---|---|---|
| `gps_glitch` | 0.907 | 0.921 |
| `motor_out` | 0.833 | 0.932 |
| `normal` | 0.965 | 0.906 |
| `wind_gust_upset` | 0.428 | 0.906 |
| `sensor_freeze` | 0.000 | 0.000 |

The synthetic-only model's 7.11% accuracy is worth sitting with: a model that scores 96.59% on synthetic test data gets almost nothing right on real flights until it is fine-tuned. Each feature added to the synthetic generator made this worse (19.69% at seven features, 7.11% at nine), because every synthetic feature distribution is one more axis on which the generator can disagree with reality. Fine-tuning on real flights closes the gap; synthetic pre-training alone does not.

No ArduPilot equivalent of PX4's public Flight Review log database exists to mine instead, so SITL generation is the only realistic real-data source for v1.

## inference_common.py / ardupilot_log.py

The evaluation/inference pipeline is split into an autopilot-agnostic half and an ArduPilot-specific half.

`inference_common.py` - `build_windows`, `temporal_split`, `evaluate_predictions`, `load_artifacts`, `predict`, `summarize_segments`, `build_timeline_figure`, `attach_ground_truth`, `extract_labeled_real_windows`. None of these read a specific log format or fault signal; they operate on whatever strings are in a `ground_truth_mode` column, and `extract_labeled_real_windows` takes its log-loading and ground-truth functions as parameters.

`ardupilot_log.py` - the ArduPilot-native half:

- `load_flight_log` reads a `.bin` dataflash log via `pymavlink.mavutil.mavlink_connection` (which auto-detects the dataflash format from the file itself) and returns the nine features listed under flight_data.py. `roll_angle`/`pitch_angle` come directly from `ATT.Roll`/`ATT.Pitch` in degrees - no quaternion math needed, simpler than PX4's `vehicle_attitude`. A message type missing from a log yields a zero-filled column rather than an exception. `vertical_speed` comes from `CTUN.CRt`; `horizontal_speed` from `GPS.Spd`. **Not yet verified against a real logged flight**: `CTUN.CRt`'s sign convention, and whether an EKF-fused velocity field would track the live MAVLink path's `LOCAL_POSITION_NED.vx/vy` better than raw GPS ground speed - both need checking once real SITL logs exist.
- `load_scripted_fault_ground_truth` reads the `<log>.fault_windows.json` sidecar that `scripts/sitl_generate_fault_logs.py` writes alongside each generated log, and labels every row `normal` or the named fault by elapsed flight time. This is the primary, exact ground truth source, and the only one for `wind_gust_upset`/`sensor_freeze` (ArduPilot has no native failsafe signal for either).
- `load_native_fault_ground_truth` reads ArduPilot's own `ERR` messages, gated on the first armed `EV` so boot-time transients aren't mislabeled as faults. The gate accepts `EV` id 10 (`ARMED`) **or** 15 (`AUTO_ARMED`) and falls back to the log's first row: real SITL logs open at the arming instant and the `ARMED` event itself never lands in the file, so gating on id 10 alone made this function return `None` on every log it was written for. An `ERR` with `ECode == 0` means resolved and now closes the fault window instead of being ignored, which previously labeled everything from the first error to the end of the flight as faulty. Only `ERR.Subsys` code 25 (`THRUST_LOSS_CHECK`) is confirmed by name.
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
- `verify_parity.py` + `parity_check.cpp` - confirms the C++ forward pass matches Keras on real windows. **Verified against `data/sitl_motor_out_1.bin`, one of this project's own generated fault logs**, on the current 18-feature models: 200 windows checked, 0 label mismatches, max probability difference 0.000000.
- `benchmark.cpp` - isolated latency measurement, needs no real log. Measured on this dev machine with the current 18-feature models over 2000 iterations: **311.68 microseconds for both models combined, 155.84 microseconds per single-model inference** - comfortably inside a 100Hz (10ms) control-loop budget. Going from 8 to 18 input features cost nothing measurable (the previous 4-feature model benchmarked at 313/156 us), because the LSTM's cost is dominated by its 64 units and 15-step window, not by the input width. This is a dev laptop, not target hardware.
- `export_replay_csv.py` + `main_replay.cpp` - the C++ equivalent of `live_inference.py --mode replay`.

`weights_current_fault.h`/`weights_next_fault.h` are generated output that can drift from the actual `.keras` models if someone retrains and forgets to re-export. The `cpp-parity` job in `.github/workflows/tests.yml` catches this on every push by regenerating the headers and failing the build if they don't match, then rebuilding and rerunning the parity check against the live models. `verify_parity.py` reads `data/sitl_motor_out_1.bin`, which is now committed, so the job has a real log to run against.

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

This now runs end to end. 25+ consecutive flights completed without intervention, producing the 15-flight dataset the models are fine-tuned on. Getting there needed four fixes, each found by running the thing rather than reading it:

- **Ordering.** The script armed first and waited for EKF convergence afterwards. On a fresh SITL boot the EKF takes 41.5 seconds to report `POS_HORIZ_ABS|POS_VERT_ABS` (measured, mostly GPS 3D-fix wait), so arming was rejected with "Need Position Estimate", the retry loop gave up after 45s, and the takeoff that followed hit `Mode::do_user_takeoff_U_m`'s plain `armed()` check on a vehicle that was never armed. This is the root cause behind the "rejection reason flaps between runs" symptom documented here previously. The sequence is now EKF → home position → GUIDED → arm → takeoff, and the first flight with that order reached 20.0 m.
- **Log rotation.** `LOG_FILE_DSRMROT` defaults to 0, meaning ArduPilot writes one dataflash log per *boot*, not per flight. `newest_bin_log` therefore returned the same still-growing file for every flight in a batch: of the first 8 logs generated, 7 were byte-for-byte the same flight, each labeled with a different fault. The sidecar for `sitl_gps_glitch_1.bin` claimed a GPS glitch at 41-56s when that stretch of the log was actually the first flight's `motor_out`. Fixed by setting `LOG_FILE_DSRMROT=1`, plus a guard that refuses to write a log identical to the previous flight's.
- **Label time base.** Fault window timestamps were measured from the start of `generate_one_flight`, but the dataflash log opens at the arming instant - 40-60 seconds later. Every label was shifted by that gap. `t0` is now the moment arming is confirmed.
- **Socket drain.** The script used `time.sleep()` through the background/hold/recover phases, reading nothing for ~55 seconds while ArduPilot kept sending at 10 Hz. Replaced with a `drain()` helper that keeps consuming messages.

Each flight randomizes altitude (10-40 m), background/hold/recover durations, ambient wind (0-6 m/s) and flies a randomly-headed velocity leg at 2-9 m/s, because the earlier fixed profile - hover at 20 m, no wind, no horizontal motion - produced flights whose mean horizontal speed was 0.16 m/s. Real flight looks nothing like that, and a model trained against it learned that any quiet moment was a frozen sensor. `--fault none` flies the same profile with no injection at all, producing the fault-free control flights that make false-alarm rate measurable.

Prerequisite for running this elsewhere: a working `ArduPilot/ardupilot` checkout with `sim_vehicle.py` set up per ArduPilot's own SITL docs.

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

The arm/takeoff ordering fix described under `sitl_generate_fault_logs.py` applies here too, and the EKF wait timeout was raised from 30s to 150s - the measured convergence time is 41.5s, so the old value could not succeed on a fresh boot.

Run (once SITL is running separately):

```bash
python scripts/sitl_live_monitor.py --connection tcp:127.0.0.1:5760
```

## Tests

Unit tests for the deterministic pieces: domain randomization, fault-segment generator shapes/trends, the ArduPilot ground-truth extraction functions (against synthetic fixtures, no real `.bin` file needed), windowing (including the horizon shift), `evaluate_predictions`, `temporal_split`, artifact loading, segment summarization, and `live_inference.py`'s buffer/advisory logic. Runs on push/PR.

```bash
pytest
```

## scripts/validate_sitl_logs.py

A gate that runs over generated logs before they are allowed into training. It exists because two dataset-corrupting bugs (the shared log file and the shifted label time base, both described above) survived unnoticed for a full generation batch - nothing was checking the data.

Per log it verifies: the sidecar parses; the log has usable rows; duration falls in 60-400s (a longer log means several flights ended up in one file); every feature stays inside a physically sane range (this is what a repeat of the 57x radians/degrees bug would trip on); every fault window falls inside the flight; the vehicle was actually above 1 m during the fault; and no two logs share a start time and duration.

```bash
python scripts/validate_sitl_logs.py data/*.bin
```

## scripts/measure_lead_time.py

Measures the number this project exists to produce: how long before ArduPilot's own detector our model names the same fault. For each log it finds the first *sustained* prediction (3 consecutive windows, so a single noisy window doesn't count) after the injection, finds ArduPilot's own `ERR` for that fault, and reports the difference.

Current result on the three `motor_out` flights:

| Log | Injected | Model | ArduPilot | Lead |
|---|---|---|---|---|
| `sitl_motor_out_1.bin` | 53.0s | 54.2s | 64.9s | 10.7s |
| `sitl_motor_out_2.bin` | 36.1s | 37.3s | 43.4s | 6.1s |
| `sitl_motor_out_3.bin` | 44.5s | 45.6s | 57.8s | 12.2s |

Mean lead: **9.7 seconds**. ArduPilot's `THRUST_LOSS_CHECK` is deliberately conservative - it triggers a failsafe, so it cannot afford false positives - which is precisely the gap a learned model can occupy: name the fault earlier, advise rather than act.

Three flights is a small sample and the numbers vary by a factor of two across them.

```bash
python scripts/measure_lead_time.py data/sitl_motor_out_*.bin
```

## scripts/evaluate_real_logs.py

Per-class precision/recall on real logs, plus the false-alarm count on fault-free control flights. Accuracy alone is misleading here: the holdout is 90.41% `normal`, so a model that predicts `normal` unconditionally scores 90.41%.

```bash
python scripts/evaluate_real_logs.py data/sitl_*.bin
```

## scripts/intervention_experiment.py

A controlled A/B experiment that asks the question a lead-time number cannot answer on its own: does warning earlier actually change what happens to the vehicle? Each trial flies the same fault twice from a freshly booted SITL - once letting ArduPilot handle it alone (`baseline`), once switching to `LAND` the moment the model raises a sustained `motor_out` alarm (`model`) - and records impact descent rate (highest descent rate in the 2 seconds before touchdown), max descent rate, distance from home, and whether ArduPilot declared a crash.

First run, 4 trials at `SIM_ENGINE_MUL=0.4` (one motor at 40% thrust), 30 m altitude:

| Arm | n | Mean impact descent | Mean distance from home | Crashes |
|---|---|---|---|---|
| baseline | 4 | 8.08 m/s | 109.6 m | 0/4 |
| model | 3 | 8.06 m/s | 109.0 m | 0/3 |

**No measurable difference.** The model raised its alarm 0.65s after injection and the intervention fired immediately, and it changed nothing. The reason is mechanical rather than statistical: with one motor at 40% thrust the vehicle is already descending uncontrolled, and `LAND` commands a descent - it cannot return thrust the vehicle does not have. Early warning is only worth as much as the action it triggers, and `LAND` is the wrong action here.

What this measures correctly and what it does not: the harness itself is now sound (single MAVLink consumer, fresh SITL per arm, force-disarm before each trial, impact rate measured over a 2-second pre-touchdown window). What it has not yet explored is the intervention (RTL, which uses remaining altitude to get closer to home, rather than LAND) or the severity range (at 40% thrust loss there may be no recoverable outcome at all; a sweep would show whether a region exists where intervening early matters).

Getting this to run took four fixes worth recording, since three of them produced plausible-looking but invalid data first: the reader thread and the flight-control functions were both calling `recv_match` on the same connection and stealing each other's `COMMAND_ACK`/`HEARTBEAT` messages; home position was captured before GPS was valid, yielding a 14,949 km distance-from-home; the vehicle stayed `armed` with `landed_state=IN_AIR` at -1.4 m between trials, so every subsequent takeoff was rejected with `MAV_RESULT=4` and no `STATUSTEXT` to explain it; and impact descent rate was sampled at the instant altitude crossed 0.5 m, by which point the vehicle had already slowed, reporting 0.0 m/s for a flight that hit the ground at 8 m/s.

```bash
python scripts/intervention_experiment.py --trials 4 --severity 0.4
```

## Explainability

`inference_common.explain_window` returns the evidence behind a prediction, not just the label. It occludes each feature in turn - replacing it and its delta with the training-set mean - and reports how far the predicted class's confidence falls. Model-agnostic, no gradients, so it ports to `cpp/` unchanged.

On a real `motor_out` flight:

```
t= 20.0s  wind_gust_upset (1.00) - kanit: roll_angle=-27.34 (tipik 1.15), horizontal_speed=6.53 (tipik 3.19)
t= 40.0s  motor_out (1.00) - kanit: motor_spread=800.00 (tipik 31.07), baro_climb_rate=-13.59 (tipik -0.08)
```

This is only possible because the features are engineered physical quantities. It also explains the failure mode: at t=20s the model says `wind_gust_upset` and its evidence is roll angle - the vehicle really was being pushed around by wind, and `wind_gust_upset` is the only label available for that.

## Tests

Unit tests for the deterministic pieces: domain randomization, fault-segment generator shapes/trends (including the new features' signatures), the ArduPilot ground-truth extraction functions and log feature units against synthetic fixtures, windowing, `evaluate_predictions`, `temporal_split`, artifact loading, segment summarization, explanation ranking, and `live_inference.py`'s buffer/advisory logic. 41 tests, run on push/PR.

```bash
pytest
```

## Future work

- **`sensor_freeze` scores 0.000 precision and 0.000 recall on real held-out flights.** This is a data-generation problem, not a model problem, and the measurements say so directly: `SIM_BARO_FREEZE` freezes the barometer, but the EKF quietly stops trusting it and falls back to GPS altitude, so the fault never reaches any fused signal the model reads. `XKF4.SH` (the EKF's own height innovation test ratio) stays at 0.000 through the entire frozen window - the autopilot doesn't consider anything wrong either. Within-window standard deviation, the obvious "flatline detector", does not separate it either (0.0007 for `sensor_freeze` vs 0.0009 for `normal` on vertical speed). The vehicle is not experiencing the fault we think we are injecting. Fixing this means changing the injection, not the model.
- **`wind_gust_upset` scores 0.428 precision** (0.906 recall) on held-out flights, and it is the main source of false alarms on fault-free flights. Same root cause, different shape: control flights are generated with 0-6 m/s ambient wind, so the vehicle genuinely is being pushed around, and the model has no label for "windy but healthy" - `wind_gust_upset` is the closest thing available. Two fixes worth measuring: feed ArduPilot's own wind estimate (`WIND` message) as a context feature so the model can represent windy-and-fine, and make the injected gust actually exceed the vehicle's control authority. At 15 m/s it does not: attitude tracking error during an injected gust is 0.01°, identical to normal flight, meaning the "fault" is a disturbance the vehicle handles comfortably.
- **The forecaster fails the acceptance gate.** 0.452 macro-F1 on real held-out flights after fine-tuning versus 0.662 for the nowcaster, so production keeps the synthetic-only forecaster and the shipped forecast is not trustworthy. The class weighting is a likely contributor - `balanced` weights push the model away from predicting `normal`, and real flights are ~90% `normal` - but that has not been tested yet.
- **Synthetic-to-real transfer degrades as features are added**: the synthetic-only model scored 19.69% on real held-out windows with seven features and 7.11% with nine, while synthetic test accuracy rose to 96.59%. Every synthetic feature distribution is another opportunity for the generator to disagree with reality. Fine-tuning recovers it (88.53%), but this means synthetic pre-training is doing much less work than the synthetic numbers suggest.
- **The lead-time result rests on three flights**, all `motor_out`, all in SITL. `gps_glitch` has no comparable measurement because ArduPilot's GPS failsafe never fired in our logs.
- **ArduPilot's `ERR.Subsys` enum is only partially mapped** - only code 25 (`THRUST_LOSS_CHECK`) is confirmed by name.
- **No public ArduPilot log database exists** to mine for real-world fault examples - SITL generation is the only realistic v1 real-data path, and everything measured here is therefore simulation-only.
- **Early detection did not change the outcome** in the first run of `scripts/intervention_experiment.py` - see that section. The lead time is real; converting it into a better outcome is not solved.
