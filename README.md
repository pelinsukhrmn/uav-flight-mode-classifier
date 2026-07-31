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

Latest run's held-out test accuracy: **BiLSTM 85.49%** (best) > LSTM 85.33% > GRU 84.07% > Conv1D 79.90% (still the weakest, as before). Flight-level 5-fold CV agreed on the same ranking: BiLSTM 85.02% ± 0.45% > LSTM 84.07% ± 0.69% > GRU 83.79% ± 0.63% > Conv1D 79.28% ± 0.79%. `transition` is the hardest class for every architecture by a wide margin (recall 0.49-0.52 vs 0.83+ for every other class) — the blended synthetic transition samples straddle two real classes by construction, so some confusion there is expected.

(These numbers aren't directly comparable to older revisions of this README: a bug was found and fixed where the `architectures` dict held live Keras layer *instances* shared across every CV fold and the final training run, instead of building fresh layers each time — meaning "independent" CV folds were silently warm-starting on each other's trained weights. Fixed by converting `architectures` to a dict of factory functions, called fresh at each use site.)

BiLSTM being the best on synthetic test accuracy carried over to the real logs this time too, after real-data fine-tuning was applied on top (see [Real-data fine-tuning](#real-data-fine-tuning) below) — see the updated ground-truth results.

Running this script also saves the best-performing model (`flight_mode_model.keras`, possibly the fine-tuned version — see below), its feature scaler (`flight_mode_scaler.joblib`), and metadata (`flight_mode_meta.joblib`) to disk for reuse, plus a second forecasting model (see [Next-mode forecasting](#next-mode-forecasting) below).

Run:

```bash
python flight_sequence_classifier.py
```

### Next-mode forecasting

Besides predicting the *current* flight mode, this script also trains a second model that forecasts the mode `PREDICTION_HORIZON = 10` timesteps ahead (about 1-2 seconds, depending on the log's sample rate), using the exact same 10-step sliding window as input — only the label changes, from `start + window_size - 1` (current mode) to `start + window_size - 1 + horizon` (mode 10 steps later). It reuses the CV-winning architecture (BiLSTM) rather than repeating the full 4-way comparison for a second task.

- Saved artifacts: `flight_mode_next_model.keras`, `flight_mode_next_scaler.joblib`, `flight_mode_next_meta.joblib` (metadata includes `horizon`).
- Test accuracy: **55.99%**, vs 85.49% for nowcasting the current mode on the same split. This gap is expected and reported honestly rather than hidden — forecasting ahead is a strictly harder problem than reporting what's already visible in the window, especially near mode transitions. Notably, `transition` itself gets 0% recall in the forecaster: by the time a transition is 10 steps from completing, the model tends to already commit to whichever mode is arriving next rather than predicting "transition," since that label describes *now*, not 10 steps from now.
- `real_log_inference.py` and `app.py` both run this second model too, reporting a forecasted next mode/confidence and, where ground truth exists that far ahead, a horizon-shifted ground-truth accuracy via `evaluate_predictions(..., horizon=...)`.

Measured on the real logs (only where ground truth exists 10 steps ahead of a window):

| Log | Next-mode accuracy | Coverage |
|---|---|---|
| `real_flight.ulg` | 63.6% | 5.1% |
| `real_flight_2.ulg` | 69.0% | 36.3% |
| `real_flight_3_vtol.ulg` | 1.8% | 18.6% |

(Same domain-gap caveat as the current-mode table below applies to the VTOL log's near-0% score.)

### Real-data fine-tuning

Ground-truth-labeled real windows are no longer just an evaluation set — the CV-winning synthetic-trained model now gets a fine-tuning pass on them, gated by an acceptance test so real data can only improve the production model, never silently regress it:

- Only `real_flight.ulg` and `real_flight_2.ulg` are used (`real_flight_3_vtol.ulg` stays evaluation-only — different vehicle domain, would pollute rather than help).
- Each log's ground-truth-covered windows are split **temporally** 70/30 (chronological, not shuffled — avoids near-duplicate overlapping windows leaking across the split): first 70% → fine-tune-train, last 30% → held out for evaluation.
- Fine-tuning batch = real fine-tune-train windows + a random stratified "replay" subsample of the original synthetic training set (up to 3x the real count) mixed in. This rehearsal keeps the 5 classes with zero real ground-truth coverage (`ascend`/`cruise`/`descend`/`transition`/`anomaly`) from being forgotten while the model adapts to the 4 real-covered classes (`hover`/`takeoff`/`land`/`rtl`).
- Low learning rate (`1e-4`), at most 15 epochs, `EarlyStopping` monitoring accuracy on the real held-out split.
- **Acceptance test**: the fine-tuned model only replaces `flight_mode_model.keras` if it (a) matches or beats the pre-fine-tune model on the real held-out set, and (b) doesn't regress synthetic test accuracy by more than 3 points. Otherwise the synthetic-only model is kept, and the script says so explicitly on the console — a claim of "improved with real data" is never made without a measured result behind it.

This run's result:

| | Real held-out accuracy | Synthetic test accuracy |
|---|---|---|
| Before fine-tuning | 83.05% (295 windows, covering `hover`/`land`/`takeoff`) | 85.49% |
| After fine-tuning | 83.05% (unchanged) | 85.74% (+0.25 points) |

Fine-tuning passed the acceptance test — no regression on either side, and a small improvement on the synthetic test set — so it **was accepted as the new production model**: the `flight_mode_model.keras` saved by this run is the fine-tuned version. The real held-out accuracy holding flat rather than jumping reflects how little headroom this small (~979-window, 3-class-covered-in-holdout) real dataset really offers — `EarlyStopping` settled on weights close to the very first epoch. The honest takeaway is that fine-tuning here mainly proved *safe* (no regression anywhere, slight upside on synthetic) rather than transformative; a bigger, more class-diverse real dataset (see the review.px4.io idea in Future Work) is what would move this number further.

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

Measured with the current model (BiLSTM, fine-tuned on real data — see [Real-data fine-tuning](#real-data-fine-tuning) above):

| Log | Ground-truth accuracy | Coverage |
|---|---|---|
| `sample.ulg` | n/a (never leaves `MANUAL`) | 0% |
| `real_flight.ulg` | 61.6% | 5.4% |
| `real_flight_2.ulg` | 69.9% | 36.1% |
| `real_flight_3_vtol.ulg` | 0.0% | 18.6% |
| `real_flight_4_poshold.ulg` | n/a (all `POSCTL`) | 0% |
| `real_flight_5_stab.ulg` | n/a (all `STAB`) | 0% |

A next-mode (forecast) accuracy table is in [Next-mode forecasting](#next-mode-forecasting) above.

`real_flight_3_vtol.ulg`'s 0% is a genuine, verified domain-gap result, not a measurement artifact: during its `AUTO_LOITER`/`AUTO_RTL` segments the aircraft is actually circling in fixed-wing mode at ~16-18 m/s horizontal speed with 27-33° of bank — nothing like the near-stationary, near-level multirotor `hover`/`rtl` the model was trained on. It mostly predicts `anomaly` there instead, which is arguably the "least wrong" label available to it, since the closest true label (`hover`/`rtl`) isn't in its vocabulary for that flight regime.

`real_flight.ulg` and `real_flight_2.ulg` accuracy (61.6%/69.9%) reflects the fine-tuned model — before fine-tuning, the same synthetic-only BiLSTM scored 59.1%/69.0% on these two logs, so real-data fine-tuning bought a small, honest improvement here rather than the multi-point swings seen in earlier README revisions when only the architecture changed.

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

Streamlit UI around the same inference pipeline as `real_log_inference.py`. Upload a `.ulg` file, or pick one of the six bundled logs (sample bench test, and the five real flights described above) from the dropdown. Shows flight duration/sample count/mean confidence, a ground-truth accuracy metric when the log has covered `nav_state` segments, the predicted mode distribution, the sensor + mode timeline, and a table of predicted segments with a CSV download button. Also shows a "Next-mode forecast" section (predicted mode ~10 steps ahead, forecast confidence, horizon-shifted ground-truth accuracy when available, and its own segment table/CSV download) when the forecaster artifacts exist.

Requires `flight_sequence_classifier.py` to have been run first (to produce the saved model files).

Run:

```bash
streamlit run app.py
```

## Tests

`tests/` has unit tests for the pure/deterministic pieces of the pipeline (domain randomization, quaternion conversion, nav_state ground-truth mapping, windowing incl. the `horizon` shift used by next-mode forecasting, `evaluate_predictions` incl. its horizon-shifted ground-truth alignment, `temporal_split`, prefixed artifact loading, segment summarization) — no `.ulg` files or trained model required (artifact loading is tested with `monkeypatch`, not real files). Runs on push/PR via `.github/workflows/tests.yml`.

```bash
pytest
```

## Future work

The real-data fine-tuning set is currently small (~979 ground-truth-labeled windows from 2 logs, covering only 4 of 9 modes). [PX4's public Flight Review database](https://review.px4.io) hosts real, publicly-shared PX4 `.ulg` logs (with full `vehicle_status`/`nav_state` telemetry) from many users and vehicles — pulling in additional logs from there, especially ones with `AUTO_MISSION`/`POSCTL`/`ACRO`/`STAB` coverage, is the most direct way to grow both the fine-tuning set and its class diversity without needing new hardware. PX4 SITL (Gazebo/jMAVSim) is a hardware-free alternative for generating additional real-PX4-state-machine logs locally.

## Setup

```bash
pip install -r requirements.txt
```
