# Sentetik uçuş üretimi için paylaşılan sabitler ve arka plan mod parametreleri.
FEATURES = ["vertical_speed", "horizontal_speed", "roll_angle", "pitch_angle",
            "motor_spread", "ekf_vel_innov", "baro_climb_rate",
            "roll_track_err", "pitch_track_err"]

MODE_CYCLE = ["hover", "takeoff", "ascend", "cruise", "rtl", "descend", "land"]

HEALTHY_MOTOR_SPREAD = (3.0, 3.0)
HEALTHY_EKF_VEL_INNOV = (0.01, 0.02)
HEALTHY_TRACK_ERR = (0.05, 0.15)
BARO_CLIMB_RATE_NOISE = 0.25

MODE_PARAMS = {
    "hover": {"vertical_speed": (0.0, 0.15), "horizontal_speed": (0.3, 0.2), "roll_angle": (0.0, 1.0), "pitch_angle": (0.0, 1.0)},
    "takeoff": {"vertical_speed": (2.0, 0.5), "horizontal_speed": (0.2, 0.15), "roll_angle": (0.0, 1.0), "pitch_angle": (2.0, 1.5)},
    "ascend": {"vertical_speed": (3.0, 0.8), "horizontal_speed": (1.0, 0.6), "roll_angle": (0.0, 2.0), "pitch_angle": (8.0, 4.0)},
    "cruise": {"vertical_speed": (0.0, 0.3), "horizontal_speed": (10.0, 2.5), "roll_angle": (5.0, 4.0), "pitch_angle": (5.0, 3.0)},
    "rtl": {"vertical_speed": (-0.5, 0.4), "horizontal_speed": (12.0, 2.5), "roll_angle": (2.0, 2.5), "pitch_angle": (6.0, 3.0)},
    "descend": {"vertical_speed": (-3.0, 0.8), "horizontal_speed": (1.0, 0.6), "roll_angle": (0.0, 2.0), "pitch_angle": (-8.0, 4.0)},
    "land": {"vertical_speed": (-1.2, 0.4), "horizontal_speed": (0.2, 0.15), "roll_angle": (0.0, 1.0), "pitch_angle": (-2.0, 1.5)},
}

for _mode in MODE_PARAMS:
    MODE_PARAMS[_mode]["motor_spread"] = HEALTHY_MOTOR_SPREAD
    MODE_PARAMS[_mode]["ekf_vel_innov"] = HEALTHY_EKF_VEL_INNOV
    MODE_PARAMS[_mode]["baro_climb_rate"] = MODE_PARAMS[_mode]["vertical_speed"]
    MODE_PARAMS[_mode]["roll_track_err"] = HEALTHY_TRACK_ERR
    MODE_PARAMS[_mode]["pitch_track_err"] = HEALTHY_TRACK_ERR

TRAVEL_MODES = ["ascend", "cruise", "rtl", "descend"]
SPEED_FEATURES = ["vertical_speed", "horizontal_speed", "baro_climb_rate"]
SPEED_SCALE_RANGE = (0.2, 1.3)
PITCH_TRIM_RANGE = (-15.0, 8.0)


def randomize_mode_params(params, speed_scale, pitch_trim, travel_modes=TRAVEL_MODES, speed_features=SPEED_FEATURES):
    randomized = {mode: dict(values) for mode, values in params.items()}
    for mode in travel_modes:
        for feature in speed_features:
            mean, std = params[mode][feature]
            randomized[mode][feature] = (mean * speed_scale, std * speed_scale)
    for mode in randomized:
        mean, std = randomized[mode]["pitch_angle"]
        randomized[mode]["pitch_angle"] = (mean + pitch_trim, std)
    return randomized
