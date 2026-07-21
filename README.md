# UAV Flight Mode Classifier

A multi-class classification model using Random Forest to predict a UAV's flight mode (hover, ascend, descend, cruise, transition) from sensor readings.

- Features: vertical speed, horizontal speed, roll angle, pitch angle
- Dataset: Synthetic (generated with numpy, based on realistic per-mode sensor ranges, with blended transition samples between adjacent modes)
- Algorithm: Random Forest Classifier

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python flight_mode_classifier.py
```
