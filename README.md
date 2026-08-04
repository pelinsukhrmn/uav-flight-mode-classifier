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

Latest run's held-out test accuracy: **BiLSTM 84.75%** (best) > LSTM 83.79% > GRU 83.31% > Conv1D 81.87% (still the weakest, as before). Flight-level 5-fold CV ranked them similarly: BiLSTM 84.72% ± 0.26% > LSTM 84.30% ± 0.67% > GRU 82.88% ± 0.87% > Conv1D 79.80% ± 1.20%. `transition` is the hardest class for every architecture by a wide margin (LSTM recall 0.40 vs 0.96+ for every other class) — the blended synthetic transition samples straddle two real classes by construction, so some confusion there is expected.

BiLSTM actually won this run's final-test-accuracy comparison, but the production model is **pinned to `LSTM` regardless of which architecture wins** (see [Real-data fine-tuning](#real-data-fine-tuning) below) - `cpp/`'s hand-written forward pass only implements a single-layer unidirectional LSTM, so shipping a BiLSTM would silently break the C++ parity check. This is a concrete instance of the "which one wins can flip between runs" variance noted above: it's exactly why the pin exists rather than trusting the comparison to keep landing on LSTM.

LSTM edging out BiLSTM here (85.08% vs 84.90%, 0.18 points) despite BiLSTM having the higher CV mean is a good illustration of single-split noise: the two architectures are statistically indistinguishable on this data (both comfortably inside each other's CV standard deviation), and the model selection code picks by final-test accuracy rather than CV mean, so which one "wins" and gets saved as production can flip between runs on the same data. This is expected variance, not a regression — worth knowing before reading too much into small architecture-ranking swings between README revisions.

(These numbers aren't directly comparable to revisions before the one that fixed a bug where the `architectures` dict held live Keras layer *instances* shared across every CV fold and the final training run, instead of building fresh layers each time — meaning "independent" CV folds were silently warm-starting on each other's trained weights. Fixed by converting `architectures` to a dict of factory functions, called fresh at each use site.)

The pinned `LSTM` model (see above) is then fine-tuned on real data (see [Real-data fine-tuning](#real-data-fine-tuning) below) before being saved as the production model — see the updated ground-truth results.

Running this script also saves the best-performing model (`models/flight_mode_model.keras`, possibly the fine-tuned version — see below), its feature scaler (`models/flight_mode_scaler.joblib`), and metadata (`models/flight_mode_meta.joblib`) to disk for reuse, plus a second forecasting model (see [Next-mode forecasting](#next-mode-forecasting) below).

Run:

```bash
python src/flight_sequence_classifier.py
```

### Next-mode forecasting

Besides predicting the *current* flight mode, this script also trains a second model that forecasts the mode `PREDICTION_HORIZON = 10` timesteps ahead (about 1-2 seconds, depending on the log's sample rate), using the exact same 10-step sliding window as input — only the label changes, from `start + window_size - 1` (current mode) to `start + window_size - 1 + horizon` (mode 10 steps later). It reuses whichever architecture won the final test-set comparison above (currently LSTM — see the note on CV-vs-final-test ranking noise there) rather than repeating the full 4-way comparison for a second task.

- Saved artifacts: `models/flight_mode_next_model.keras`, `models/flight_mode_next_scaler.joblib`, `models/flight_mode_next_meta.joblib` (metadata includes `horizon`).
- Test accuracy: **55.99%**, vs 83.79% for nowcasting the current mode on the same split. This gap is expected and reported honestly rather than hidden — forecasting ahead is a strictly harder problem than reporting what's already visible in the window, especially near mode transitions. Notably, `transition` itself gets low recall in the forecaster: by the time a transition is 10 steps from completing, the model tends to already commit to whichever mode is arriving next rather than predicting "transition," since that label describes *now*, not 10 steps from now.
- `real_log_inference.py` and `app.py` both run this second model too, reporting a forecasted next mode/confidence and, where ground truth exists that far ahead, a horizon-shifted ground-truth accuracy via `evaluate_predictions(..., horizon=...)`.
- Unlike the nowcasting model, **the forecaster is not part of the real-data fine-tuning pass** ([Real-data fine-tuning](#real-data-fine-tuning) below) - it's trained on synthetic data only and saved directly. This is a real, current limitation, not an oversight being glossed over: see `real_flight_11_descend.ulg`'s 8.7% forecast score in the table below for what that costs on a class the forecaster has literally never seen real examples of.

Measured on the real logs (only where ground truth exists 10 steps ahead of a window):

| Log | Next-mode accuracy | Coverage |
|---|---|---|
| `real_flight.ulg` | 74.0% | 5.1% (154 windows) |
| `real_flight_2.ulg` | 67.6% | 36.3% (815 windows) |
| `real_flight_3_vtol.ulg` | 0.0% | 18.6% (3662 windows) |
| `real_flight_6_takeoff_land.ulg` | 83.9% | 100.0% (3722 windows) |
| `real_flight_7_takeoff_land.ulg` | 0.0% | 100.0% (2274 windows) |
| `real_flight_8_hover_rtl.ulg` | 96.2% | 75.7% (3846 windows) |
| `real_flight_9_hover_land.ulg` | 98.0% | 96.2% (6023 windows) |
| `real_flight_10_sitl_hover_rtl.ulg` | 66.8% | 60.7% (11939 windows) |
| `real_flight_11_descend.ulg` | 8.7% | 100.0% (3359 windows) |

(Same domain-gap caveat as the current-mode table below applies to the VTOL log's 0.0% score.)

`real_flight_11_descend.ulg`'s 8.7% is a checked result, not a fluke: on its 2616 true-`descend` forecast windows, the forecaster predicts `cruise` 68% of the time and `ascend` 21% of the time, `descend` itself only 3% of the time. Because the forecaster never gets the real-data fine-tuning pass the nowcaster does (see above), it's making this call purely from synthetic `descend`/`cruise`/`ascend` boundaries - which apparently don't separate cleanly from this real log's actual vertical-speed profile 10 steps ahead, even though the *nowcasting* model (which did get fine-tuned on this same log's `descend` windows) scores 60.3% on it. That gap between the two models on the same log is itself the clearest evidence that fine-tuning, not architecture, is what's carrying `descend` performance right now.

`real_flight_7_takeoff_land.ulg`'s 0.0% forecast score is a real, checked result, not a bug — and a different failure mode than the VTOL domain gap. For most of this flight PX4's `nav_state` stays in `AUTO_TAKEOFF` (it never transitions to `AUTO_LOITER`) even though the vehicle is physically just sitting still at altitude for over 3 minutes. The forecaster reads the physics correctly and predicts `hover` for essentially the entire flight (2284/2284 windows) — which is the right call for what the sensors show — but `nav_state` ground truth says `takeoff` throughout, so it's scored as 0% wrong against a label that itself doesn't reflect what the aircraft is actually doing. The current-mode (nowcasting) model shows the same pattern less starkly (93.8% ground-truth accuracy, because a chunk of its `takeoff` predictions do land on genuine climb segments) but its predicted mode is also mostly `hover`/`takeoff` for the stationary stretch. This is a label-quality artifact of this specific log's mission configuration, not a modeling error — worth knowing if using this log as a benchmark.

### Real-data fine-tuning

Ground-truth-labeled real windows are no longer just an evaluation set — the CV-winning synthetic-trained model now gets a fine-tuning pass on them, gated by an acceptance test so real data can only improve the production model, never silently regress it:

- `real_flight.ulg`, `real_flight_2.ulg`, four public logs pulled from [PX4's Flight Review database](https://review.px4.io) (`real_flight_6_takeoff_land.ulg` through `real_flight_9_hover_land.ulg`), a fifth public log added specifically for `descend` coverage (`real_flight_11_descend.ulg` — see below), and a PX4 SITL+Gazebo simulated flight (`real_flight_10_sitl_hover_rtl.ulg` — see below) are used (`real_flight_3_vtol.ulg`, `real_flight_4_poshold.ulg`, and `real_flight_5_stab.ulg` stay evaluation-only: different vehicle domain or no ground-truth coverage at all).
- Each log's ground-truth-covered windows are split **temporally** 70/30 (chronological, not shuffled — avoids near-duplicate overlapping windows leaking across the split): first 70% → fine-tune-train, last 30% → held out for evaluation.
- Fine-tuning batch = real fine-tune-train windows + a random stratified "replay" subsample of the original synthetic training set (up to 3x the real count) mixed in. This rehearsal keeps the 4 classes with zero real ground-truth coverage (`ascend`/`cruise`/`transition`/`anomaly`) from being forgotten while the model adapts to the 5 real-covered classes (`hover`/`takeoff`/`land`/`rtl`/`descend`).
- Low learning rate (`1e-4`), at most 15 epochs, `EarlyStopping` monitoring accuracy on the real held-out split.
- **Acceptance test**: the fine-tuned model only replaces `models/flight_mode_model.keras` if it (a) matches or beats the pre-fine-tune model on the real held-out set, and (b) doesn't regress synthetic test accuracy by more than 3 points. Otherwise the synthetic-only model is kept, and the script says so explicitly on the console — a claim of "improved with real data" is never made without a measured result behind it.
- `best_name` (which architecture gets fine-tuned and shipped) is **pinned to `LSTM`** rather than picked by max test accuracy, because `cpp/lstm_model.hpp` only implements a single-layer unidirectional LSTM forward pass — a BiLSTM/GRU/Conv1D win in the 4-way comparison above would otherwise break the `cpp/` parity check. BiLSTM has in fact won the final-test-accuracy comparison on at least one run (see the note on CV-vs-final-test ranking noise above) — the pin exists so architecture selection doesn't silently pick something `cpp/` can't run.

Result from the current fine-tuning run (8 logs — 22,524 ground-truth fine-tune-train windows, held-out set covering `hover`/`land`/`rtl`/`takeoff`/`descend`):

| | Real held-out accuracy | Synthetic test accuracy |
|---|---|---|
| Before fine-tuning | 40.58% (9658 windows) | 83.79% |
| After fine-tuning | 47.28% (+6.70 points) | 81.46% (-2.32 points) |

Fine-tuning passed the acceptance test (real accuracy improved, synthetic regression under the 3-point tolerance) — it **was accepted as the new production model**. Both the accuracy level and the size of the move are different from the previous 6-log run (73.63% → 83.48%) for a concrete reason, not noise: this run added `descend` as a real-covered class for the first time (via `real_flight_11_descend.ulg` and incidentally `real_flight_10_sitl_hover_rtl.ulg`'s own ground truth), and the pre-fine-tune model's `descend` predictions on real data started from a much weaker place than its already-tuned `hover`/`takeoff`/`land`/`rtl` — pulling the aggregate baseline down to 40.58% and leaving more room for fine-tuning to close.

A previous run of this same fine-tuning step crashed here with a `train_test_split` `ValueError`: the "replay" subsample size is capped at `min(len(synthetic_train), 3 * len(real_train))`, and with 11,802 real windows the `3x` term exceeded the entire synthetic training set for the first time (it never had with only ~979 real windows), so the code asked `train_test_split` for a test split equal in size to the whole array, which isn't a valid split. Fixed in `flight_sequence_classifier.py` by using the whole synthetic training set directly (skipping the split) whenever the cap doesn't bind.

Worth calling out honestly: `real_flight_10_sitl_hover_rtl.ulg` alone now contributes 8364 of the 22,524 fine-tune-train windows (37%) — more than any single real-hardware log — because a ~157s SITL flight sampled at PX4's native rate produces far more 10-step sliding windows than a similarly-short public log. It's simulated data (perfect Gazebo physics, no real sensor noise), not hardware, so its outsized weight in the mix is a real domain-composition tradeoff, not just a size number — flagged here in the same spirit as the earlier note on `real_flight_2.ulg`'s shrinking relative weight as more logs were added. The aggregate real held-out number is still the trustworthy summary metric; it isn't being inflated by treating simulated windows as equivalent evidence to hardware windows, just mixed in as an additional, honestly-labeled source (see the `app.py` dropdown, which marks it "simulated" explicitly).

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
| `DESCEND` | `descend` |

`DESCEND` (distinct from `AUTO_LAND` - "descend mode, no position control", typically entered via an RC-loss failsafe) was added to this mapping after finding real PX4 logs with meaningful coverage of it - unlike `AUTO_MISSION`, it's unambiguous, so it was safe to add without needing the sub-leg inference described in [Future work](#future-work).

Everything else (`MANUAL`, `ALTCTL`, `POSCTL`, `STAB`, `ACRO`, `OFFBOARD`, `AUTO_MISSION`, ...) is left unmapped on purpose — `AUTO_MISSION` alone could be `ascend`/`cruise`/`descend`/`hover` depending on which leg of the mission it is, and the manual modes have no equivalent in our label set at all. The reported `coverage` is how much of the flight fell into a mapped state; accuracy is only computed over that covered portion, so a low-coverage flight (e.g. one flown mostly by hand) is reported honestly as "not much to validate here" rather than silently padded with guesses.

Measured with the current model (LSTM, fine-tuned on real data — see [Real-data fine-tuning](#real-data-fine-tuning) above):

| Log | Ground-truth accuracy | Coverage |
|---|---|---|
| `sample.ulg` | n/a (never leaves `MANUAL`) | 0% |
| `real_flight.ulg` | 68.3% | 5.4% (164 windows) |
| `real_flight_2.ulg` | 51.2% | 36.1% (815 windows) |
| `real_flight_3_vtol.ulg` | 45.0% | 18.6% (3662 windows) |
| `real_flight_4_poshold.ulg` | n/a (all `POSCTL`) | 0% |
| `real_flight_5_stab.ulg` | n/a (all `STAB`) | 0% |
| `real_flight_6_takeoff_land.ulg` | 77.4% | 100.0% (3732 windows) |
| `real_flight_7_takeoff_land.ulg` | 90.1% | 100.0% (2284 windows) |
| `real_flight_8_hover_rtl.ulg` | 95.1% | 75.5% (3846 windows) |
| `real_flight_9_hover_land.ulg` | 83.5% | 96.1% (6023 windows) |
| `real_flight_10_sitl_hover_rtl.ulg` | 68.2% | 60.7% (11949 windows) |
| `real_flight_11_descend.ulg` | 60.3% | 100.0% (3369 windows) |

A next-mode (forecast) accuracy table is in [Next-mode forecasting](#next-mode-forecasting) above (including a note on why `real_flight_7_takeoff_land.ulg` scores 0% there specifically — a label-quality artifact of that log, not a modeling failure).

`real_flight_3_vtol.ulg`'s number moved from a previously-reported 0.0% to 45.0% after this round of retraining, and it's worth being precise about why rather than reading it as "the model learned VTOL": during its `AUTO_LOITER`/`AUTO_RTL` segments the aircraft is still actually circling in fixed-wing mode at ~16-18 m/s horizontal speed with 27-33° of bank — nothing like the near-stationary, near-level multirotor `hover`/`rtl` the model was trained on, and that hasn't changed. What changed is *which* wrong-domain label the retrained model reaches for: on the `hover`-labeled portion (3577 of 3662 covered windows) it now predicts `hover` 46% of the time and `cruise` 44% of the time, versus mostly `anomaly` before. That's close to a coin flip between two plausible-looking labels for an out-of-distribution input, not newly-learned VTOL competence — the retrained decision boundary simply shifted (most likely from `real_flight_10_sitl_hover_rtl.ulg`'s RTL-transit windows broadening what the model associates with `hover`/`cruise`) and this log's score moved as a side effect. Still a genuine domain-gap case, just a noisier one to read now.

The public multirotor logs (`real_flight_6` through `real_flight_9`, plus the new `real_flight_11_descend.ulg`) score well (60.3-95.1%) — expected, since they're close to the flight profiles (clean multirotor takeoff/hover/land/rtl/descend cycles) the fine-tuning set is dominated by. `real_flight_10_sitl_hover_rtl.ulg` (68.2%) is simulated rather than hardware — see [Real-data fine-tuning](#real-data-fine-tuning) above for why it still counts as real ground truth despite being sim-sourced, and for the caveat about its large weight in the fine-tuning mix. See that section too for why `real_flight.ulg`/`real_flight_2.ulg` individually moved rather than both improving — the aggregate real held-out number there is the more meaningful summary than any single log's accuracy.

`data/sample.ulg` is a real PX4 log (from the [pyulog](https://github.com/PX4/pyulog) test suite) used as the example input. Note: this particular log is a stationary bench/attitude test, not an actual flight (near-zero velocity, but large roll swings from the vehicle being manually rotated) — depending on the trained model, it may not predict `hover` throughout, which is a real example of the sim-to-real gap: the synthetic training data never paired large attitude changes with near-zero velocity, since normal flight doesn't produce that combination. Its `nav_state` never leaves `MANUAL`, so it has no ground-truth coverage at all.

Eleven genuine flights are also included:

- `data/real_flight.ulg` — a ~10-minute multirotor flight, low horizontal speed throughout (0-3.5 m/s) and a persistent ~-12 deg pitch trim even at hover. Against the original `MODE_PARAMS` (single fixed speed profile, hover pitch centered on 0), this log broke the model in two different, instructive ways: first it predicted `transition` almost everywhere (no travel mode's speed range was ever close to this slow), and after the speed-scale fix alone, it flipped to predicting `descend` almost everywhere (its constant negative pitch trim looked exactly like the `descend` class' pitch mean once slow-mode speeds started overlapping hover). Only after also randomizing pitch trim per flight did the predicted timeline start tracking the actual altitude changes (e.g. the sustained descent in the last ~60s of the flight is correctly predicted as `descend`). Ground truth covers its `AUTO_LOITER`/`AUTO_TAKEOFF`/`AUTO_LAND` segments.
- `data/real_flight_2.ulg` — a shorter (~3.5 min), faster, near-zero-trim multirotor flight (horizontal speed up to 14 m/s). It validates the other end of the widened range: the two fast horizontal bursts early in the flight are correctly picked up as `cruise`/`rtl` instead of being invisible to the model.
- `data/real_flight_3_vtol.ulg` — a ~33-minute VTOL (fixed-wing-capable) test flight, flown mostly in manual `STAB`/`ACRO` modes with only brief `AUTO_LOITER`/`AUTO_RTL`/`AUTO_MISSION` segments. This is a different domain-gap example from the other two: the model was only ever trained on a multirotor hover→takeoff→ascend→cruise→rtl→descend→land mission cycle, so a mostly hand-flown VTOL log is expected to have low ground-truth coverage and is included as an honest stress test, not a validated success case.
- `data/real_flight_4_poshold.ulg` — a ~2.4-minute multirotor flight flown entirely in `POSCTL` (manual position hold). No ground-truth coverage by design (`POSCTL` isn't in the mapping table above — a pilot can translate freely in position hold, so it has no single equivalent label), but it's a real, continuous manual-flight example for judging the timeline by eye.
- `data/real_flight_5_stab.ulg` — a ~1-minute multirotor flight flown entirely in `STAB` (manual stabilized). Same story as above: no ground-truth coverage, included as another real hand-flown segment.
- `data/real_flight_6_takeoff_land.ulg` through `data/real_flight_9_hover_land.ulg` — four public quadrotor logs pulled from [PX4's Flight Review database](https://review.px4.io) (`review.px4.io/dbinfo` + the CDN download URL each entry provides), chosen from ~1300 candidates filtered for a "good"/"great" community rating, zero logged errors, and strong `hover`/`takeoff`/`land`/`rtl` ground-truth coverage (76-100% of each flight). Added specifically to grow the real-data fine-tuning set (see below) beyond just 2 source flights, without needing new hardware. Different vehicles/airframes than `real_flight.ulg`/`real_flight_2.ulg`, which is the point — it reduces how much the fine-tuned model is shaped by any one aircraft's quirks.
- `data/real_flight_10_sitl_hover_rtl.ulg` — a ~157s flight flown entirely in PX4 SITL + Gazebo (`gz_x500` quadrotor model), not on real hardware: a scripted MAVSDK mission (arm → 4-waypoint `AUTO_MISSION` loop → paused mid-mission for an explicit `AUTO_LOITER` hold → resumed → `AUTO_RTL` back to launch). Ground truth covers `hover` (140 windows, from the scripted hold) and `rtl` (62 windows); the `AUTO_MISSION` legs are unmapped for the same reason real hardware `AUTO_MISSION` legs are. Included as the hardware-free option the [Future work](#future-work) section previously flagged as unexplored — it's simulated, not hardware, so it's labeled as such in `app.py`'s dropdown and weighted into that context in the fine-tuning discussion above rather than presented as equivalent to a real flight.
- `data/real_flight_11_descend.ulg` — a public quadrotor log chosen specifically for its `DESCEND` nav_state coverage (530 of 690 ground-truth windows, plus `hover`/`takeoff`/`rtl`) once the `DESCEND` → `descend` mapping (see [Ground-truth accuracy](#ground-truth-accuracy-not-just-eyeballing-the-timeline) above) made that state usable. The first real ground-truth source for the `descend` class, which previously had none.

Requires `src/flight_sequence_classifier.py` to have been run first (to produce the saved model files).

Run:

```bash
python src/flight_sequence_classifier.py
python src/real_log_inference.py data/real_flight.ulg
```

## app.py

Streamlit UI around the same inference pipeline as `real_log_inference.py`. Upload a `.ulg` file, or pick one of the twelve bundled logs (sample bench test, and the eleven real flights described above) from the dropdown. Shows flight duration/sample count/mean confidence, a ground-truth accuracy metric when the log has covered `nav_state` segments, the predicted mode distribution, the sensor + mode timeline, and a table of predicted segments with a CSV download button. Also shows a "Next-mode forecast" section (predicted mode ~10 steps ahead, forecast confidence, horizon-shifted ground-truth accuracy when available, and its own segment table/CSV download) when the forecaster artifacts exist.

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
- **`verify_parity.py`** + **`parity_check.cpp`** — the correctness check that actually matters: dumps real windows and the genuine Keras models' output probabilities, then confirms the C++ forward pass reproduces them. Checked on 200 real windows (100 current-mode + 100 next-mode, from `real_flight_2.ulg`): **0 label mismatches, max probability difference 0.000001** — indistinguishable from float32 rounding noise, not an approximation.
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

~~The real-data fine-tuning set is currently small (~979 ground-truth-labeled windows from 2 logs, covering only 4 of 9 modes).~~ **Done**: 4 more public logs from PX4's Flight Review database were added (see `real_flight_6`-`real_flight_9` above), growing real ground-truth coverage roughly 6x.

~~PX4 SITL (Gazebo/jMAVSim) remains a hardware-free option for generating more real-PX4-state-machine logs locally in the meantime.~~ **Done**: `real_flight_10_sitl_hover_rtl.ulg` was generated with a scripted MAVSDK mission against PX4 SITL + Gazebo (`gz_x500`) and folded into the fine-tuning set - see above for its coverage and the honest caveat about its outsized share of the fine-tuning windows (simulated, not hardware).

~~Actually adding coverage for the other 5 classes (`ascend`/`cruise`/`descend`/`transition`/`anomaly`) would need a different approach than "find more public logs".~~ **Partially done for `descend`**: PX4's `nav_state` turned out to have an unambiguous `DESCEND` state (distinct from `AUTO_MISSION`, see the mapping table above) that public logs do carry meaningful coverage of - no sub-leg inference needed, just a missing mapping entry. `ascend`/`cruise` remain unsolved for the reason originally stated: they only ever appear as sub-legs of `AUTO_MISSION`, which PX4 doesn't record as a distinct nav_state, so getting real ground truth for them would need inferring sub-legs from waypoint/altitude-setpoint data - a real design problem, not just a data-collection one. `transition` (a synthetic construct - blended samples between adjacent modes) likely has no real equivalent to collect at all. `anomaly` remains 100% synthetic; real logs with genuine faults (logged errors, vibration, sensor-error tags) exist in PX4's Flight Review database and were identified as candidates during this round of work (e.g. a log with 824 logged errors tagged `Sensor-error`+`Other`), but turning "this log has some logged errors somewhere" into per-window `anomaly` ground truth needs its own labeling approach - `vehicle_status.nav_state` has no `anomaly` state to map from, unlike `descend` - and wasn't attempted here to keep this round of work scoped.

`live_inference.py` and `cpp/` (above) are deliberately advisory-only, not a controller wired into the autopilot. Closing that loop would need: real ground-truth coverage for the `anomaly` class (currently 100% synthetic — the one class where being wrong matters most), a measured/bounded inference latency budget on the *actual target hardware* (the ~3-4 microsecond/inference number in `cpp/` is from this dev machine, not a companion computer like a Raspberry Pi or Jetson — still unmeasured there, and MAVLink message parsing/buffering overhead isn't included either), and a case for why a probabilistic classifier should override PX4's own deterministic mode state machine at all, when it already handles mode transitions reliably. None of that exists yet, so the safe scope for now is exactly what's implemented: printing forecasts a human or a downstream advisory system can act on.
