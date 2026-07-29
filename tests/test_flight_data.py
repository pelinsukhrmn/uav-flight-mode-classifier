from flight_data import MODE_PARAMS, TRAVEL_MODES, randomize_mode_params


def test_speed_scale_only_applies_to_travel_modes():
    randomized = randomize_mode_params(MODE_PARAMS, speed_scale=0.5, pitch_trim=0.0)

    for mode in TRAVEL_MODES:
        mean, std = MODE_PARAMS[mode]["vertical_speed"]
        r_mean, r_std = randomized[mode]["vertical_speed"]
        assert r_mean == mean * 0.5
        assert r_std == std * 0.5

    non_travel_modes = [m for m in MODE_PARAMS if m not in TRAVEL_MODES]
    for mode in non_travel_modes:
        assert randomized[mode]["vertical_speed"] == MODE_PARAMS[mode]["vertical_speed"]
        assert randomized[mode]["horizontal_speed"] == MODE_PARAMS[mode]["horizontal_speed"]


def test_pitch_trim_applies_to_every_mode():
    randomized = randomize_mode_params(MODE_PARAMS, speed_scale=1.0, pitch_trim=-12.0)

    for mode in MODE_PARAMS:
        mean, std = MODE_PARAMS[mode]["pitch_angle"]
        r_mean, r_std = randomized[mode]["pitch_angle"]
        assert r_mean == mean - 12.0
        assert r_std == std


def test_original_params_are_not_mutated():
    before = {mode: dict(values) for mode, values in MODE_PARAMS.items()}
    randomize_mode_params(MODE_PARAMS, speed_scale=0.3, pitch_trim=5.0)
    assert MODE_PARAMS == before
