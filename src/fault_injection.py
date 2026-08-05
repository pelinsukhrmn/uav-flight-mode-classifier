# Sentetik arıza (motor kaybı, GPS glitch, rüzgar sarsıntısı, sensör donması) segment üreticileri.
import numpy as np

from flight_data import FEATURES

FAULT_CLASSES = ["motor_out", "gps_glitch", "wind_gust_upset", "sensor_freeze"]


def sample_motor_out_segment(n, rng, background_params):
    onset = 1.0 - np.exp(-np.linspace(0.0, 4.0, n))
    roll_sign = rng.choice([-1.0, 1.0])
    roll_peak = rng.uniform(15.0, 40.0)
    pitch_peak = rng.choice([-1.0, 1.0]) * rng.uniform(10.0, 25.0)
    vspeed_drop = rng.uniform(1.5, 4.0)

    vspeed_mean, vspeed_std = background_params["vertical_speed"]
    hspeed_mean, hspeed_std = background_params["horizontal_speed"]
    roll_mean, roll_std = background_params["roll_angle"]
    pitch_mean, pitch_std = background_params["pitch_angle"]

    return {
        "vertical_speed": vspeed_mean - onset * vspeed_drop + rng.normal(0.0, vspeed_std, n),
        "horizontal_speed": np.abs(rng.normal(hspeed_mean, hspeed_std, n)),
        "roll_angle": roll_mean + onset * roll_sign * roll_peak + rng.normal(0.0, roll_std, n),
        "pitch_angle": pitch_mean + onset * pitch_peak + rng.normal(0.0, pitch_std, n),
    }


def sample_gps_glitch_segment(n, rng, background_params):
    step = rng.choice([-1.0, 1.0]) * rng.uniform(5.0, 15.0)
    decay = np.exp(-np.linspace(0.0, 3.0, n))

    vspeed_mean, vspeed_std = background_params["vertical_speed"]
    hspeed_mean, hspeed_std = background_params["horizontal_speed"]
    roll_mean, roll_std = background_params["roll_angle"]
    pitch_mean, pitch_std = background_params["pitch_angle"]

    return {
        "vertical_speed": rng.normal(vspeed_mean, vspeed_std, n),
        "horizontal_speed": np.abs(hspeed_mean + step * decay + rng.normal(0.0, hspeed_std * 1.5, n)),
        "roll_angle": rng.normal(roll_mean, roll_std, n),
        "pitch_angle": rng.normal(pitch_mean, pitch_std, n),
    }


def sample_wind_gust_upset_segment(n, rng, background_params):
    gust_shape = np.sin(np.linspace(0.0, np.pi, n)) ** 0.5
    recovery_fraction = rng.uniform(0.4, 0.8)
    fade = 1.0 - recovery_fraction * (np.arange(n) / max(n - 1, 1))
    roll_sign = rng.choice([-1.0, 1.0])
    roll_peak = rng.uniform(10.0, 25.0)
    hspeed_peak = rng.uniform(4.0, 12.0)

    vspeed_mean, vspeed_std = background_params["vertical_speed"]
    hspeed_mean, hspeed_std = background_params["horizontal_speed"]
    roll_mean, roll_std = background_params["roll_angle"]
    pitch_mean, pitch_std = background_params["pitch_angle"]

    return {
        "vertical_speed": rng.normal(vspeed_mean, vspeed_std, n),
        "horizontal_speed": np.abs(hspeed_mean + gust_shape * hspeed_peak * fade + rng.normal(0.0, hspeed_std, n)),
        "roll_angle": roll_mean + gust_shape * roll_sign * roll_peak * fade + rng.normal(0.0, roll_std, n),
        "pitch_angle": rng.normal(pitch_mean, pitch_std, n),
    }


def sample_sensor_freeze_segment(n, rng, background_params, frozen_feature=None):
    frozen_feature = frozen_feature or rng.choice(FEATURES)
    out = {}
    for feature in FEATURES:
        mean, std = background_params[feature]
        if feature == frozen_feature:
            out[feature] = np.full(n, rng.normal(mean, std))
        else:
            out[feature] = rng.normal(mean, std, n)
    if frozen_feature != "horizontal_speed":
        out["horizontal_speed"] = np.abs(out["horizontal_speed"])
    return out


FAULT_GENERATORS = {
    "motor_out": sample_motor_out_segment,
    "gps_glitch": sample_gps_glitch_segment,
    "wind_gust_upset": sample_wind_gust_upset_segment,
    "sensor_freeze": sample_sensor_freeze_segment,
}
