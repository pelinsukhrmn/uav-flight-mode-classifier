# Sentetik arıza (motor kaybı, GPS glitch, rüzgar sarsıntısı, sensör donması) segment üreticileri.
import numpy as np

from flight_data import FEATURES, BARO_CLIMB_RATE_NOISE

FAULT_CLASSES = ["motor_out", "gps_glitch", "wind_gust_upset", "sensor_freeze"]
FREEZABLE_FEATURES = ["vertical_speed", "horizontal_speed", "roll_angle", "pitch_angle", "baro_climb_rate"]
MOTOR_OUT_SPREAD_RANGE = (450.0, 750.0)
MOTOR_OUT_TRACK_ERR_RANGE = (5.0, 60.0)
GPS_GLITCH_INNOV_RANGE = (0.8, 2.0)


def healthy_extras(n, rng, background_params, vertical_speed):
    spread_mean, spread_std = background_params["motor_spread"]
    innov_mean, innov_std = background_params["ekf_vel_innov"]
    roll_err_mean, roll_err_std = background_params["roll_track_err"]
    pitch_err_mean, pitch_err_std = background_params["pitch_track_err"]
    return {
        "motor_spread": np.abs(rng.normal(spread_mean, spread_std, n)),
        "ekf_vel_innov": np.abs(rng.normal(innov_mean, innov_std, n)),
        "baro_climb_rate": vertical_speed + rng.normal(0.0, BARO_CLIMB_RATE_NOISE, n),
        "roll_track_err": np.abs(rng.normal(roll_err_mean, roll_err_std, n)),
        "pitch_track_err": np.abs(rng.normal(pitch_err_mean, pitch_err_std, n)),
    }


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

    vertical_speed = vspeed_mean - onset * vspeed_drop + rng.normal(0.0, vspeed_std, n)
    out = {
        "vertical_speed": vertical_speed,
        "horizontal_speed": np.abs(rng.normal(hspeed_mean, hspeed_std, n)),
        "roll_angle": roll_mean + onset * roll_sign * roll_peak + rng.normal(0.0, roll_std, n),
        "pitch_angle": pitch_mean + onset * pitch_peak + rng.normal(0.0, pitch_std, n),
    }
    out.update(healthy_extras(n, rng, background_params, vertical_speed))
    out["motor_spread"] = onset * rng.uniform(*MOTOR_OUT_SPREAD_RANGE) + np.abs(rng.normal(0.0, 20.0, n))
    out["roll_track_err"] = onset * rng.uniform(*MOTOR_OUT_TRACK_ERR_RANGE) + np.abs(rng.normal(0.0, 1.0, n))
    out["pitch_track_err"] = onset * rng.uniform(*MOTOR_OUT_TRACK_ERR_RANGE) + np.abs(rng.normal(0.0, 1.0, n))
    return out


def sample_gps_glitch_segment(n, rng, background_params):
    step = rng.choice([-1.0, 1.0]) * rng.uniform(5.0, 15.0)
    decay = np.exp(-np.linspace(0.0, 3.0, n))

    vspeed_mean, vspeed_std = background_params["vertical_speed"]
    hspeed_mean, hspeed_std = background_params["horizontal_speed"]
    roll_mean, roll_std = background_params["roll_angle"]
    pitch_mean, pitch_std = background_params["pitch_angle"]

    vertical_speed = rng.normal(vspeed_mean, vspeed_std, n)
    out = {
        "vertical_speed": vertical_speed,
        "horizontal_speed": np.abs(hspeed_mean + step * decay + rng.normal(0.0, hspeed_std * 1.5, n)),
        "roll_angle": rng.normal(roll_mean, roll_std, n),
        "pitch_angle": rng.normal(pitch_mean, pitch_std, n),
    }
    out.update(healthy_extras(n, rng, background_params, vertical_speed))
    out["ekf_vel_innov"] = np.abs(rng.uniform(*GPS_GLITCH_INNOV_RANGE) + rng.normal(0.0, 0.3, n))
    return out


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

    vertical_speed = rng.normal(vspeed_mean, vspeed_std, n)
    out = {
        "vertical_speed": vertical_speed,
        "horizontal_speed": np.abs(hspeed_mean + gust_shape * hspeed_peak * fade + rng.normal(0.0, hspeed_std, n)),
        "roll_angle": roll_mean + gust_shape * roll_sign * roll_peak * fade + rng.normal(0.0, roll_std, n),
        "pitch_angle": rng.normal(pitch_mean, pitch_std, n),
    }
    out.update(healthy_extras(n, rng, background_params, vertical_speed))
    return out


def sample_sensor_freeze_segment(n, rng, background_params, frozen_feature=None):
    frozen_feature = frozen_feature or rng.choice(FREEZABLE_FEATURES)
    out = {}
    for feature in FEATURES:
        mean, std = background_params[feature]
        out[feature] = rng.normal(mean, std, n)

    out["motor_spread"] = np.abs(out["motor_spread"])
    out["ekf_vel_innov"] = np.abs(out["ekf_vel_innov"])
    out["roll_track_err"] = np.abs(out["roll_track_err"])
    out["pitch_track_err"] = np.abs(out["pitch_track_err"])
    out["baro_climb_rate"] = out["vertical_speed"] + rng.normal(0.0, BARO_CLIMB_RATE_NOISE, n)

    if frozen_feature == "baro_climb_rate":
        out["baro_climb_rate"] = np.zeros(n)
    else:
        mean, std = background_params[frozen_feature]
        out[frozen_feature] = np.full(n, rng.normal(mean, std))

    if frozen_feature != "horizontal_speed":
        out["horizontal_speed"] = np.abs(out["horizontal_speed"])
    return out


FAULT_GENERATORS = {
    "motor_out": sample_motor_out_segment,
    "gps_glitch": sample_gps_glitch_segment,
    "wind_gust_upset": sample_wind_gust_upset_segment,
    "sensor_freeze": sample_sensor_freeze_segment,
}
