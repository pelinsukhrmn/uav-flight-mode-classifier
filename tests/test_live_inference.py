import pandas as pd
import pytest

from live_inference import LiveWindowBuffer, advisory


META = {"features": ["v", "v_delta"], "window_size": 3}


def make_row(t, v):
    return {"timestamp": t, "v": v}


def test_live_window_buffer_empty_until_full():
    buffer = LiveWindowBuffer(META)
    for t in range(META["window_size"]):  # window_size rows: not window_size + 1 yet
        buffer.push(make_row(t, float(t)))
        window, end_ts = buffer.latest_window()
        assert window is None
        assert end_ts is None


def test_live_window_buffer_produces_window_matching_offline_deltas():
    buffer = LiveWindowBuffer(META)
    values = [1.0, 2.0, 4.0, 7.0]  # window_size + 1 rows
    for t, v in enumerate(values):
        buffer.push(make_row(t, v))
    window, end_ts = buffer.latest_window()

    assert window is not None
    assert window.shape == (1, META["window_size"], len(META["features"]))
    assert end_ts == META["window_size"]  # last row's timestamp

    offline = pd.DataFrame({"timestamp": range(len(values)), "v": values})
    offline["v_delta"] = offline["v"].diff().fillna(0)
    expected = offline[META["features"]].to_numpy()[1:]  # rows 1..3, matching the buffered window
    assert (window[0] == expected).all()


def test_advisory_none_when_forecast_matches_current():
    assert advisory("hover", 0.9, "hover", 0.9, horizon_seconds=1.0) is None


def test_advisory_flags_urgent_transitions():
    msg = advisory("hover", 0.9, "land", 0.8, horizon_seconds=2.0)
    assert msg is not None
    assert "UYARI" in msg


def test_advisory_informational_for_non_urgent_transitions():
    msg = advisory("hover", 0.9, "cruise", 0.7, horizon_seconds=2.0)
    assert msg is not None
    assert "bilgi" in msg
