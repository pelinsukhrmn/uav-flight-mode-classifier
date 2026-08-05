# fault_injection.py'nin sentetik arıza segment üreticilerini şekil/eğilim bazında test eder.
import numpy as np
import pytest

from fault_injection import (
    sample_motor_out_segment, sample_gps_glitch_segment,
    sample_wind_gust_upset_segment, sample_sensor_freeze_segment,
)

BACKGROUND_PARAMS = {
    "vertical_speed": (0.0, 0.15),
    "horizontal_speed": (10.0, 2.5),
    "roll_angle": (5.0, 4.0),
    "pitch_angle": (5.0, 3.0),
}


def test_motor_out_roll_diverges_monotonically_outward():
    rng = np.random.default_rng(0)
    segment = sample_motor_out_segment(50, rng, BACKGROUND_PARAMS)
    deviation = np.abs(segment["roll_angle"] - BACKGROUND_PARAMS["roll_angle"][0])
    smoothed = np.convolve(deviation, np.ones(5) / 5, mode="valid")
    assert smoothed[-1] > smoothed[0]


def test_motor_out_vertical_speed_drops_below_background():
    rng = np.random.default_rng(1)
    segment = sample_motor_out_segment(50, rng, BACKGROUND_PARAMS)
    assert segment["vertical_speed"][-5:].mean() < BACKGROUND_PARAMS["vertical_speed"][0]


def test_gps_glitch_produces_discontinuity_from_background():
    rng = np.random.default_rng(2)
    segment = sample_gps_glitch_segment(50, rng, BACKGROUND_PARAMS)
    jump = abs(segment["horizontal_speed"][0] - BACKGROUND_PARAMS["horizontal_speed"][0])
    assert jump > BACKGROUND_PARAMS["horizontal_speed"][1]


def test_wind_gust_upset_partially_recovers_by_segment_end():
    rng = np.random.default_rng(3)
    segment = sample_wind_gust_upset_segment(60, rng, BACKGROUND_PARAMS)
    peak_deviation = np.abs(segment["roll_angle"] - BACKGROUND_PARAMS["roll_angle"][0]).max()
    end_deviation = np.abs(segment["roll_angle"][-5:].mean() - BACKGROUND_PARAMS["roll_angle"][0])
    assert 0 < end_deviation < peak_deviation


def test_sensor_freeze_flatlines_only_the_chosen_feature():
    rng = np.random.default_rng(4)
    segment = sample_sensor_freeze_segment(30, rng, BACKGROUND_PARAMS, frozen_feature="roll_angle")
    assert np.all(segment["roll_angle"] == segment["roll_angle"][0])
    assert segment["pitch_angle"].std() > 0
