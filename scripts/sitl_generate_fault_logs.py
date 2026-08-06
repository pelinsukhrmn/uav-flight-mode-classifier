# ArduPilot SITL'de kontrollü arıza enjekte edip etiketli .bin log + fault_windows.json üretir.
import argparse
import json
import math
import random
import shutil
import threading
import time
from pathlib import Path

from pymavlink import mavutil


def heartbeat_loop(conn):
    while True:
        conn.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(1.0)

FAULT_PARAMS = {
    "motor_out": [("SIM_ENGINE_FAIL", 1), ("SIM_ENGINE_MUL", 0.0)],
    "gps_glitch": [("SIM_GPS1_VERR_X", 8.0), ("SIM_GPS1_VERR_Y", 8.0)],
    "wind_gust_upset": [("SIM_WIND_SPD", 15.0), ("SIM_WIND_DIR", 90.0), ("SIM_WIND_TC", 2.0)],
    "sensor_freeze": [("SIM_BARO_FREEZE", 1)],
}
FAULT_CLEAR = {
    "motor_out": [("SIM_ENGINE_FAIL", 0), ("SIM_ENGINE_MUL", 1.0)],
    "gps_glitch": [("SIM_GPS1_VERR_X", 0.0), ("SIM_GPS1_VERR_Y", 0.0)],
    "wind_gust_upset": [("SIM_WIND_SPD", 0.0)],
    "sensor_freeze": [("SIM_BARO_FREEZE", 0)],
}


def set_param(conn, name, value):
    conn.mav.param_set_send(conn.target_system, conn.target_component, name.encode(), float(value), mavutil.mavlink.MAV_PARAM_TYPE_REAL32)


def wait_mode(conn, mode_name, timeout=30):
    mode_id = conn.mode_mapping()[mode_name]
    conn.set_mode(mode_id)
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = conn.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
        if msg is not None and msg.custom_mode == mode_id:
            return True
    return False


def arm_with_retry(conn, timeout=45, resend_interval=3):
    deadline = time.time() + timeout
    last_send = 0.0
    while time.time() < deadline:
        if time.time() - last_send >= resend_interval:
            conn.mav.command_long_send(
                conn.target_system, conn.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0,
            )
            last_send = time.time()
        msg = conn.recv_match(type=["HEARTBEAT", "COMMAND_ACK", "STATUSTEXT"], blocking=True, timeout=1)
        if msg is None:
            continue
        msg_type = msg.get_type()
        if msg_type == "HEARTBEAT" and bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
            return True
        if msg_type == "COMMAND_ACK" and msg.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM and msg.result != 0:
            print(f"  arm rejected: MAV_RESULT={msg.result}, retrying", flush=True)
        if msg_type == "STATUSTEXT":
            print(f"  STATUSTEXT: {msg.text}", flush=True)
    return False


def drain(conn, duration_s):
    deadline = time.time() + duration_s
    while time.time() < deadline:
        conn.recv_match(blocking=True, timeout=0.5)


def wait_disarmed(conn, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = conn.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        if msg is not None and not bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
            return True
    return False


def wait_for_ekf_position_ok(conn, timeout=150):
    required = mavutil.mavlink.EKF_POS_HORIZ_ABS | mavutil.mavlink.EKF_POS_VERT_ABS
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = conn.recv_match(type="EKF_STATUS_REPORT", blocking=True, timeout=1)
        if msg is not None and (msg.flags & required) == required:
            return True
    return False


def wait_for_home_position(conn, timeout=30, resend_interval=3):
    deadline = time.time() + timeout
    last_send = 0.0
    while time.time() < deadline:
        if time.time() - last_send >= resend_interval:
            conn.mav.command_long_send(
                conn.target_system, conn.target_component,
                mavutil.mavlink.MAV_CMD_GET_HOME_POSITION, 0, 0, 0, 0, 0, 0, 0, 0,
            )
            last_send = time.time()
        msg = conn.recv_match(type="HOME_POSITION", blocking=True, timeout=1)
        if msg is not None:
            return True
    return False


def takeoff_with_retry(conn, altitude, timeout=20, resend_interval=3):
    deadline = time.time() + timeout
    last_send = 0.0
    while time.time() < deadline:
        if time.time() - last_send >= resend_interval:
            conn.mav.command_long_send(
                conn.target_system, conn.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, altitude,
            )
            last_send = time.time()
        msg = conn.recv_match(type=["COMMAND_ACK", "STATUSTEXT"], blocking=True, timeout=1)
        if msg is None:
            continue
        msg_type = msg.get_type()
        if msg_type == "COMMAND_ACK" and msg.command == mavutil.mavlink.MAV_CMD_NAV_TAKEOFF:
            print(f"  takeoff ack: MAV_RESULT={msg.result}", flush=True)
            if msg.result == 0:
                return True
        if msg_type == "STATUSTEXT":
            print(f"  STATUSTEXT: {msg.text}", flush=True)
    return False


def arm_and_takeoff(conn, altitude, climb_wait_s):
    if not wait_for_ekf_position_ok(conn):
        print("  WARNING: EKF position never reported ok - attempting arm anyway", flush=True)
    if not wait_for_home_position(conn):
        print("  WARNING: home position never confirmed - attempting arm anyway", flush=True)

    wait_mode(conn, "GUIDED")
    if not arm_with_retry(conn):
        print("  WARNING: never confirmed armed - proceeding anyway, flight will likely be invalid", flush=True)
    armed_wall = time.time()

    if not takeoff_with_retry(conn, altitude):
        print("  WARNING: takeoff command never ACKed as accepted - proceeding anyway", flush=True)

    climb_deadline = time.time() + climb_wait_s
    while time.time() < climb_deadline:
        msg = conn.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        if msg is not None and not bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
            print("  WARNING: disarmed during climb wait", flush=True)
    return armed_wall


def inject_fault(conn, fault_type, hold_s, rng):
    for name, value in FAULT_PARAMS[fault_type]:
        set_param(conn, name, value)
    fly_random_leg(conn, rng, hold_s)
    for name, value in FAULT_CLEAR[fault_type]:
        set_param(conn, name, value)


def fly_velocity(conn, vx, vy, duration_s, resend_interval=0.5):
    type_mask = 0b0000_11_111_000_111
    deadline = time.time() + duration_s
    last_send = 0.0
    while time.time() < deadline:
        if time.time() - last_send >= resend_interval:
            conn.mav.set_position_target_local_ned_send(
                0, conn.target_system, conn.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED, type_mask,
                0, 0, 0, vx, vy, 0, 0, 0, 0, 0, 0,
            )
            last_send = time.time()
        conn.recv_match(blocking=True, timeout=0.2)


def fly_random_leg(conn, rng, duration_s):
    speed = rng.uniform(2.0, 9.0)
    heading = rng.uniform(0, 2 * math.pi)
    fly_velocity(conn, speed * math.cos(heading), speed * math.sin(heading), duration_s)


def newest_bin_log(log_dir, after_mtime):
    candidates = [p for p in Path(log_dir).glob("*.BIN") if p.stat().st_mtime > after_mtime]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def generate_one_flight(conn, fault_type, out_dir, log_dir, index, rng, previous_log_path=None, climb_wait_s=15.0):
    start_mtime = time.time()

    altitude = rng.uniform(10.0, 40.0)
    background_s = rng.uniform(20.0, 40.0)
    hold_s = rng.uniform(12.0, 25.0)
    recover_s = rng.uniform(12.0, 25.0)
    ambient_wind = rng.uniform(0.0, 2.0) if fault_type == "wind_gust_upset" else rng.uniform(0.0, 6.0)
    set_param(conn, "SIM_WIND_SPD", ambient_wind)
    set_param(conn, "SIM_WIND_DIR", rng.uniform(0.0, 360.0))
    print(f"  alt={altitude:.0f}m background={background_s:.0f}s hold={hold_s:.0f}s ambient_wind={ambient_wind:.1f}m/s", flush=True)

    log_t0_wall = arm_and_takeoff(conn, altitude, climb_wait_s)
    fly_random_leg(conn, rng, background_s)

    windows = []
    if fault_type != "none":
        fault_start_s = time.time() - log_t0_wall
        inject_fault(conn, fault_type, hold_s, rng)
        windows.append({"fault": fault_type, "start_s": fault_start_s, "end_s": time.time() - log_t0_wall})
        set_param(conn, "SIM_WIND_SPD", ambient_wind)

    fly_random_leg(conn, rng, recover_s)
    wait_mode(conn, "RTL")
    if not wait_disarmed(conn):
        print("  WARNING: vehicle never disarmed after RTL - log may be truncated", flush=True)

    drain(conn, 2.0)
    log_path = newest_bin_log(log_dir, start_mtime)
    if log_path is None:
        print(f"  no new .BIN found in {log_dir} - copy it manually", flush=True)
        return None
    if previous_log_path is not None and log_path == previous_log_path:
        print(f"  {log_path} is the same log as the previous flight - skipping (check LOG_FILE_DSRMROT)", flush=True)
        return log_path

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_bin = Path(out_dir) / f"sitl_{fault_type}_{index}.bin"
    out_json = Path(out_dir) / f"sitl_{fault_type}_{index}.fault_windows.json"
    shutil.copy(log_path, out_bin)
    out_json.write_text(json.dumps(windows))
    print(f"  wrote {out_bin} + {out_json}", flush=True)
    return log_path


def main():
    parser = argparse.ArgumentParser(description="Generate labeled ArduPilot SITL fault logs.")
    parser.add_argument("--fault", required=True, choices=list(FAULT_PARAMS) + ["none"])
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--connection", default="udp:127.0.0.1:14550")
    parser.add_argument("--sitl-log-dir", required=True, help="ArduPilot SITL's logs/ directory")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    conn = mavutil.mavlink_connection(args.connection)
    conn.wait_heartbeat()
    conn.mav.request_data_stream_send(conn.target_system, conn.target_component, mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1)
    threading.Thread(target=heartbeat_loop, args=(conn,), daemon=True).start()
    set_param(conn, "DISARM_DELAY", 0)
    set_param(conn, "LOG_FILE_DSRMROT", 1)

    rng = random.Random(args.seed)
    previous_log_path = None
    for i in range(args.count):
        index = args.start_index + i
        print(f"Flight {i + 1}/{args.count}: injecting {args.fault} (index {index})", flush=True)
        previous_log_path = generate_one_flight(
            conn, args.fault, args.out_dir, args.sitl_log_dir, index, rng, previous_log_path,
        ) or previous_log_path


if __name__ == "__main__":
    main()
