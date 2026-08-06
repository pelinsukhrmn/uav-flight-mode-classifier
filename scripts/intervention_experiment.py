# Ayni arizayi iki kolda ucurup erken mudahalenin sonucunu olcen kontrollu deney.
import argparse
import json
import math
import queue
import random
import statistics
import sys
import threading
import time
from collections import deque
from pathlib import Path

from pymavlink import mavutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_inference import MavlinkFeatureState, LiveWindowBuffer, score_window
from inference_common import load_artifacts
from sitl_generate_fault_logs import fly_velocity, heartbeat_loop, set_param

SUSTAIN_WINDOWS = 3
IMPACT_WINDOW_S = 2.0
ALARM_FAULTS = ("motor_out",)


class TrialState:
    def __init__(self):
        self.lock = threading.Lock()
        self.armed = False
        self.relative_alt_m = 0.0
        self.vertical_speed = 0.0
        self.distance_from_home_m = 0.0
        self.max_descent_rate = 0.0
        self.impact_descent_rate = None
        self.descent_history = deque()
        self.crashed = False
        self.alarm_time = None
        self.statustexts = []
        self.custom_mode = None
        self.last_arm_ack = None
        self.last_takeoff_ack = None
        self.ekf_flags = 0
        self.home_set = False
        self.landed_state = None


def reader_loop(conn, meta, state, window_queue, stop_event):
    features = MavlinkFeatureState()
    buffer = LiveWindowBuffer(meta)
    home = None
    last_scored_ts = None

    while not stop_event.is_set():
        msg = conn.recv_match(blocking=True, timeout=1)
        if msg is None:
            continue
        msg_type = msg.get_type()

        if msg_type == "HEARTBEAT":
            with state.lock:
                state.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                state.custom_mode = msg.custom_mode
            continue
        if msg_type == "COMMAND_ACK":
            with state.lock:
                if msg.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
                    state.last_arm_ack = msg.result
                elif msg.command == mavutil.mavlink.MAV_CMD_NAV_TAKEOFF:
                    state.last_takeoff_ack = msg.result
            continue
        if msg_type == "EKF_STATUS_REPORT":
            with state.lock:
                state.ekf_flags = msg.flags
            continue
        if msg_type == "EXTENDED_SYS_STATE":
            with state.lock:
                state.landed_state = msg.landed_state
            continue
        if msg_type == "STATUSTEXT":
            with state.lock:
                state.statustexts.append(msg.text)
                if "crash" in msg.text.lower():
                    state.crashed = True
            continue
        if msg_type == "HOME_POSITION":
            home = (msg.latitude, msg.longitude)
            with state.lock:
                state.home_set = True
            continue
        if msg_type == "GLOBAL_POSITION_INT":
            if msg.lat == 0 and msg.lon == 0:
                continue
            with state.lock:
                state.relative_alt_m = msg.relative_alt / 1000.0
                state.vertical_speed = msg.vz / -100.0
                if state.vertical_speed < 0:
                    state.max_descent_rate = max(state.max_descent_rate, -state.vertical_speed)
                now = time.time()
                if state.relative_alt_m > 0.5:
                    state.descent_history.append((now, -min(state.vertical_speed, 0.0)))
                    while state.descent_history and now - state.descent_history[0][0] > IMPACT_WINDOW_S:
                        state.descent_history.popleft()
                elif state.impact_descent_rate is None and state.descent_history:
                    state.impact_descent_rate = max(rate for _, rate in state.descent_history)
                if home is not None:
                    state.distance_from_home_m = haversine_m(home, (msg.lat, msg.lon))
            continue

        row = features.update(msg)
        if row is None:
            continue
        buffer.push(row)
        window, end_ts = buffer.latest_window()
        if window is None:
            continue
        if last_scored_ts is not None and (end_ts - last_scored_ts) / 1e6 < 0.2:
            continue
        last_scored_ts = end_ts
        try:
            window_queue.put_nowait(window)
        except queue.Full:
            pass


def wait_state(state, predicate, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with state.lock:
            if predicate(state):
                return True
        time.sleep(0.2)
    return False


def set_mode(conn, state, mode_name, timeout=20):
    mode_id = conn.mode_mapping()[mode_name]
    deadline = time.time() + timeout
    last_send = 0.0
    while time.time() < deadline:
        if time.time() - last_send >= 2:
            conn.set_mode(mode_id)
            last_send = time.time()
        with state.lock:
            if state.custom_mode == mode_id:
                return True
        time.sleep(0.2)
    return False


def wait_for_ekf(conn, state, timeout=180):
    required = mavutil.mavlink.EKF_POS_HORIZ_ABS | mavutil.mavlink.EKF_POS_VERT_ABS
    return wait_state(state, lambda s: (s.ekf_flags & required) == required, timeout)


def wait_for_home(conn, state, timeout=60):
    deadline = time.time() + timeout
    last_send = 0.0
    while time.time() < deadline:
        if time.time() - last_send >= 3:
            conn.mav.command_long_send(
                conn.target_system, conn.target_component,
                mavutil.mavlink.MAV_CMD_GET_HOME_POSITION, 0, 0, 0, 0, 0, 0, 0, 0,
            )
            last_send = time.time()
        with state.lock:
            if state.home_set:
                return True
        time.sleep(0.2)
    return False


def force_disarm_and_ground(conn, state, timeout=90):
    deadline = time.time() + timeout
    last_send = 0.0
    while time.time() < deadline:
        with state.lock:
            armed = state.armed
            landed = state.landed_state
        if not armed and landed in (None, mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND):
            return True
        if time.time() - last_send >= 3:
            conn.mav.command_long_send(
                conn.target_system, conn.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 0, 21196, 0, 0, 0, 0, 0,
            )
            last_send = time.time()
        time.sleep(0.2)
    return False


def arm_vehicle(conn, state, timeout=60):
    deadline = time.time() + timeout
    last_send = 0.0
    while time.time() < deadline:
        if time.time() - last_send >= 3:
            conn.mav.command_long_send(
                conn.target_system, conn.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0,
            )
            last_send = time.time()
        with state.lock:
            if state.armed:
                return True
        time.sleep(0.2)
    return False


def takeoff(conn, state, altitude, timeout=30):
    deadline = time.time() + timeout
    last_send = 0.0
    while time.time() < deadline:
        if time.time() - last_send >= 3:
            conn.mav.command_long_send(
                conn.target_system, conn.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, altitude,
            )
            last_send = time.time()
        with state.lock:
            ack = state.last_takeoff_ack
            alt = state.relative_alt_m
        if ack == 0 or alt > 1.0:
            return True
        time.sleep(0.2)
    return False


def disarm_and_settle(conn, state, timeout=120):
    set_mode(conn, state, "LAND", timeout=10)
    if not wait_state(state, lambda s: not s.armed, timeout):
        conn.mav.command_long_send(
            conn.target_system, conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 0, 21196, 0, 0, 0, 0, 0,
        )
        wait_state(state, lambda s: not s.armed, 30)
    time.sleep(3)


def haversine_m(a, b):
    lat1, lon1 = a[0] / 1e7, a[1] / 1e7
    lat2, lon2 = b[0] / 1e7, b[1] / 1e7
    r = 6371000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    h = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def scorer_loop(window_queue, scaler, model, meta, state, stop_event, alarm_start_wall):
    run = 0
    while not stop_event.is_set():
        try:
            window = window_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        label, _ = score_window(window, scaler, model, meta)
        run = run + 1 if label in ALARM_FAULTS else 0
        if run >= SUSTAIN_WINDOWS:
            with state.lock:
                if state.alarm_time is None:
                    state.alarm_time = time.time() - alarm_start_wall
            run = 0


def reboot_autopilot(conn, connection_string, wait_s=8):
    try:
        conn.mav.command_long_send(
            conn.target_system, conn.target_component,
            mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN, 0, 1, 0, 0, 0, 0, 0, 0,
        )
    except Exception:
        pass
    conn.close()
    time.sleep(wait_s)
    for _ in range(30):
        try:
            fresh = mavutil.mavlink_connection(connection_string)
            fresh.wait_heartbeat(timeout=10)
            if fresh.target_system:
                return fresh
        except Exception:
            time.sleep(2)
    raise RuntimeError("autopilot did not come back after reboot")


def prepare_connection(conn):
    conn.mav.request_data_stream_send(conn.target_system, conn.target_component,
                                      mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1)
    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
        mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE_TARGET, 100000, 0, 0, 0, 0, 0,
    )
    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
        mavutil.mavlink.MAVLINK_MSG_ID_EXTENDED_SYS_STATE, 500000, 0, 0, 0, 0, 0,
    )
    threading.Thread(target=heartbeat_loop, args=(conn,), daemon=True).start()
    set_param(conn, "DISARM_DELAY", 0)
    set_param(conn, "LOG_FILE_DSRMROT", 1)


def run_trial(conn, meta, scaler, model, arm, severity, altitude, rng, settle_s, observe_s):
    state = TrialState()
    stop_event = threading.Event()
    window_queue = queue.Queue(maxsize=8)

    set_param(conn, "SIM_WIND_SPD", 0.0)
    set_param(conn, "SIM_ENGINE_FAIL", 0)
    set_param(conn, "SIM_ENGINE_MUL", 1.0)

    reader = threading.Thread(target=reader_loop, args=(conn, meta, state, window_queue, stop_event), daemon=True)
    reader.start()

    if not force_disarm_and_ground(conn, state):
        with state.lock:
            landed = state.landed_state
        stop_event.set()
        return {"arm": arm, "valid": False, "reason": f"vehicle stuck armed/in-air (landed_state={landed})"}
    if not wait_for_ekf(conn, state):
        stop_event.set()
        return {"arm": arm, "valid": False, "reason": "EKF never converged"}
    wait_for_home(conn, state)
    if not set_mode(conn, state, "GUIDED"):
        stop_event.set()
        return {"arm": arm, "valid": False, "reason": "never entered GUIDED"}
    if not arm_vehicle(conn, state):
        stop_event.set()
        return {"arm": arm, "valid": False, "reason": "never armed"}
    if not takeoff(conn, state, altitude):
        with state.lock:
            ack = state.last_takeoff_ack
            texts = state.statustexts[-3:]
        stop_event.set()
        return {"arm": arm, "valid": False, "reason": f"takeoff rejected (ack={ack}, {texts})"}

    wait_state(state, lambda s: s.relative_alt_m >= altitude * 0.8, 60)
    with state.lock:
        if state.relative_alt_m < altitude * 0.5:
            stop_event.set()
            return {"arm": arm, "valid": False, "reason": f"never climbed (alt={state.relative_alt_m:.1f}m)"}

    heading = rng.uniform(0, 2 * math.pi)
    fly_velocity(conn, 5.0 * math.cos(heading), 5.0 * math.sin(heading), settle_s)

    while not window_queue.empty():
        try:
            window_queue.get_nowait()
        except queue.Empty:
            break
    injection_wall = time.time()
    scorer = threading.Thread(
        target=scorer_loop, args=(window_queue, scaler, model, meta, state, stop_event, injection_wall), daemon=True,
    )
    scorer.start()

    set_param(conn, "SIM_ENGINE_FAIL", 1)
    set_param(conn, "SIM_ENGINE_MUL", severity)

    intervened_at = None
    deadline = time.time() + observe_s
    while time.time() < deadline:
        with state.lock:
            armed = state.armed
            alarm_time = state.alarm_time
        if not armed:
            break
        if arm == "model" and alarm_time is not None and intervened_at is None:
            set_mode(conn, state, "LAND", timeout=10)
            intervened_at = alarm_time
        time.sleep(0.2)

    with state.lock:
        result = {
            "arm": arm,
            "valid": True,
            "alarm_time_s": state.alarm_time,
            "intervened_at_s": intervened_at,
            "crashed": state.crashed,
            "max_descent_rate_ms": round(state.max_descent_rate, 2),
            "impact_descent_rate_ms": None if state.impact_descent_rate is None else round(state.impact_descent_rate, 2),
            "distance_from_home_m": round(state.distance_from_home_m, 1),
            "final_alt_m": round(state.relative_alt_m, 1),
            "still_armed": state.armed,
        }

    set_param(conn, "SIM_ENGINE_FAIL", 0)
    set_param(conn, "SIM_ENGINE_MUL", 1.0)
    disarm_and_settle(conn, state)
    stop_event.set()
    time.sleep(1.0)
    return result


def summarize(results):
    by_arm = {}
    for result in results:
        if result.get("valid"):
            by_arm.setdefault(result["arm"], []).append(result)
    print("\n=== SUMMARY ===")
    for arm, rows in sorted(by_arm.items()):
        crashes = sum(1 for r in rows if r["crashed"])
        impacts = [r["impact_descent_rate_ms"] for r in rows if r["impact_descent_rate_ms"] is not None]
        distances = [r["distance_from_home_m"] for r in rows]
        print(f"{arm:10s} n={len(rows)}  crashes={crashes}/{len(rows)}  "
              f"mean impact descent={statistics.mean(impacts) if impacts else float('nan'):.2f} m/s  "
              f"mean distance from home={statistics.mean(distances):.1f} m")


def main():
    parser = argparse.ArgumentParser(description="A/B experiment: autopilot alone vs model-triggered early LAND.")
    parser.add_argument("--connection", default="tcp:127.0.0.1:5760")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--severity", type=float, default=0.4, help="SIM_ENGINE_MUL during the fault")
    parser.add_argument("--altitude", type=float, default=30.0)
    parser.add_argument("--settle-s", type=float, default=20.0)
    parser.add_argument("--observe-s", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/intervention_results.json")
    parser.add_argument("--arm", choices=["baseline", "model"], help="run a single arm and exit")
    parser.add_argument("--trial-index", type=int, default=0)
    parser.add_argument("--no-reboot", action="store_true")
    args = parser.parse_args()

    meta, scaler, model = load_artifacts("fault")
    conn = mavutil.mavlink_connection(args.connection)
    conn.wait_heartbeat()
    conn.mav.request_data_stream_send(conn.target_system, conn.target_component, mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1)
    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
        mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE_TARGET, 100000, 0, 0, 0, 0, 0,
    )
    prepare_connection(conn)

    if args.arm:
        rng = random.Random(args.seed + args.trial_index)
        result = run_trial(conn, meta, scaler, model, args.arm, args.severity, args.altitude, rng,
                           args.settle_s, args.observe_s)
        result["trial"] = args.trial_index
        print(f"    {result}", flush=True)
        existing = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else []
        existing.append(result)
        Path(args.out).write_text(json.dumps(existing, indent=2))
        summarize(existing)
        return

    results = []
    for trial in range(args.trials):
        for arm in ("baseline", "model"):
            rng = random.Random(args.seed + trial)
            if not args.no_reboot:
                conn = reboot_autopilot(conn, args.connection)
                prepare_connection(conn)
            print(f"\n--- trial {trial + 1}/{args.trials}, arm={arm}, severity={args.severity}", flush=True)
            result = run_trial(conn, meta, scaler, model, arm, args.severity, args.altitude, rng,
                               args.settle_s, args.observe_s)
            result["trial"] = trial
            if not result.get("valid"):
                print(f"    INVALID: {result.get('reason')} - retrying once", flush=True)
                time.sleep(5)
                rng = random.Random(args.seed + trial)
                conn = reboot_autopilot(conn, args.connection)
                prepare_connection(conn)
                result = run_trial(conn, meta, scaler, model, arm, args.severity, args.altitude, rng,
                                   args.settle_s, args.observe_s)
                result["trial"] = trial
            print(f"    {result}", flush=True)
            results.append(result)
            Path(args.out).write_text(json.dumps(results, indent=2))

    summarize(results)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
