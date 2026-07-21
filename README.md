# UAV Flight Mode Classifier

Two models that predict a UAV's flight mode from sensor readings (vertical speed, horizontal speed, roll angle, pitch angle).

## flight_mode_classifier.py

Multi-class classification on independent sensor readings (hover, ascend, descend, cruise, transition).

- Dataset: Synthetic (generated with numpy, based on realistic per-mode sensor ranges, with blended transition samples between adjacent modes)
- Algorithm: Random Forest Classifier

Run:

```bash
python flight_mode_classifier.py
```

## flight_sequence_classifier.py

Time-series classification over sliding windows of consecutive sensor readings, with an added `anomaly` class simulating sensor malfunction or erratic control loss.

- Dataset: Synthetic continuous flight logs (120 simulated flights, each cycling through hover, ascend, cruise, descend with smooth transitions, and occasional injected anomaly segments)
- Features: the 4 raw sensor readings plus their timestep-to-timestep delta (rate of change), 8 features total
- Algorithm: LSTM (Keras/TensorFlow), trained on 10-timestep sliding windows
- Train/test split is done by flight, not by window, to avoid data leakage between overlapping windows
- Class weighting (`sklearn.utils.class_weight`) is used to counter the rarity of the `anomaly` class

Run:

```bash
python flight_sequence_classifier.py
```

## Setup

```bash
pip install -r requirements.txt
```
