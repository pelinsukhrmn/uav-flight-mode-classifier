FEATURES = ["vertical_speed", "horizontal_speed", "roll_angle", "pitch_angle"]

MODE_CYCLE = ["hover", "takeoff", "ascend", "cruise", "rtl", "descend", "land"]

MODE_PARAMS = {
    "hover": {"vertical_speed": (0.0, 0.15), "horizontal_speed": (0.3, 0.2), "roll_angle": (0.0, 1.0), "pitch_angle": (0.0, 1.0)},
    "takeoff": {"vertical_speed": (2.0, 0.5), "horizontal_speed": (0.2, 0.15), "roll_angle": (0.0, 1.0), "pitch_angle": (2.0, 1.5)},
    "ascend": {"vertical_speed": (3.0, 0.8), "horizontal_speed": (1.0, 0.6), "roll_angle": (0.0, 2.0), "pitch_angle": (8.0, 4.0)},
    "cruise": {"vertical_speed": (0.0, 0.3), "horizontal_speed": (10.0, 2.5), "roll_angle": (5.0, 4.0), "pitch_angle": (5.0, 3.0)},
    "rtl": {"vertical_speed": (-0.5, 0.4), "horizontal_speed": (12.0, 2.5), "roll_angle": (2.0, 2.5), "pitch_angle": (6.0, 3.0)},
    "descend": {"vertical_speed": (-3.0, 0.8), "horizontal_speed": (1.0, 0.6), "roll_angle": (0.0, 2.0), "pitch_angle": (-8.0, 4.0)},
    "land": {"vertical_speed": (-1.2, 0.4), "horizontal_speed": (0.2, 0.15), "roll_angle": (0.0, 1.0), "pitch_angle": (-2.0, 1.5)},
}

TRAVEL_MODES = ["ascend", "cruise", "rtl", "descend"]
SPEED_FEATURES = ["vertical_speed", "horizontal_speed"]
SPEED_SCALE_RANGE = (0.2, 1.3)

# Real airframes can hover/fly with a persistent non-zero pitch trim (rigging,
# CG offset) that has nothing to do with flight mode. A real PX4 log showed a
# ~-12 deg pitch bias throughout the whole flight, which a pitch_angle mean of
# 0 for "hover" can't represent. Randomizing a per-flight trim teaches the
# model to rely on pitch relative to the flight's own baseline.
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
