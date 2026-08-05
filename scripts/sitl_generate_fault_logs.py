# ArduPilot SITL'de kontrollü arıza enjekte edip etiketli .bin log + fault_windows.json üretir.
import argparse
import json
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
            print(f"  arm rejected: MAV_RESULT={msg.result}, retrying")
        if msg_type == "STATUSTEXT":
            print(f"  STATUSTEXT: {msg.text}")
    return False


def wait_for_ekf_position_ok(conn, timeout=30):
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
            print(f"  takeoff ack: MAV_RESULT={msg.result}")
            if msg.result == 0:
                return True
        if msg_type == "STATUSTEXT":
            print(f"  STATUSTEXT: {msg.text}")
    return False


def arm_and_takeoff(conn, altitude, climb_wait_s):
    wait_mode(conn, "GUIDED")
    if not arm_with_retry(conn):
        print("  WARNING: never confirmed armed - proceeding anyway, flight will likely be invalid")
    if not wait_for_ekf_position_ok(conn):
        print("  WARNING: EKF position never reported ok - attempting takeoff anyway")
    if not wait_for_home_position(conn):
        print("  WARNING: home position never confirmed - attempting takeoff anyway")

    print("  settling 8s after arming before takeoff (re-arming if needed)")
    settle_deadline = time.time() + 8
    while time.time() < settle_deadline:
        msg = conn.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        if msg is not None and not bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
            print("  re-arming during settle wait...")
            arm_with_retry(conn, timeout=10)

    if not takeoff_with_retry(conn, altitude):
        print("  WARNING: takeoff command never ACKed as accepted - proceeding anyway")

    climb_deadline = time.time() + climb_wait_s
    while time.time() < climb_deadline:
        msg = conn.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        if msg is not None and not bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
            print("  WARNING: disarmed during climb wait")


def inject_fault(conn, fault_type, hold_s):
    for name, value in FAULT_PARAMS[fault_type]:
        set_param(conn, name, value)
    time.sleep(hold_s)
    for name, value in FAULT_CLEAR[fault_type]:
        set_param(conn, name, value)


def newest_bin_log(log_dir, after_mtime):
    candidates = [p for p in Path(log_dir).glob("*.BIN") if p.stat().st_mtime > after_mtime]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def generate_one_flight(conn, fault_type, out_dir, log_dir, index,
                         altitude=20.0, climb_wait_s=15.0, background_s=25.0, hold_s=15.0, recover_s=15.0):
    start_mtime = time.time()
    flight_start_wall = time.time()

    arm_and_takeoff(conn, altitude, climb_wait_s)
    time.sleep(background_s)

    fault_start_s = time.time() - flight_start_wall
    inject_fault(conn, fault_type, hold_s)
    fault_end_s = time.time() - flight_start_wall

    time.sleep(recover_s)
    wait_mode(conn, "RTL")
    conn.motors_disarmed_wait()

    time.sleep(2.0)
    log_path = newest_bin_log(log_dir, start_mtime)
    if log_path is None:
        print(f"  no new .BIN found in {log_dir} - copy it manually")
        return

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_bin = Path(out_dir) / f"sitl_{fault_type}_{index}.bin"
    out_json = Path(out_dir) / f"sitl_{fault_type}_{index}.fault_windows.json"
    shutil.copy(log_path, out_bin)
    out_json.write_text(json.dumps([{"fault": fault_type, "start_s": fault_start_s, "end_s": fault_end_s}]))
    print(f"  wrote {out_bin} + {out_json}")


def main():
    parser = argparse.ArgumentParser(description="Generate labeled ArduPilot SITL fault logs.")
    parser.add_argument("--fault", required=True, choices=list(FAULT_PARAMS))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--connection", default="udp:127.0.0.1:14550")
    parser.add_argument("--sitl-log-dir", required=True, help="ArduPilot SITL's logs/ directory")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--start-index", type=int, default=1)
    args = parser.parse_args()

    conn = mavutil.mavlink_connection(args.connection)
    conn.wait_heartbeat()
    threading.Thread(target=heartbeat_loop, args=(conn,), daemon=True).start()

    for i in range(args.count):
        index = args.start_index + i
        print(f"Flight {i + 1}/{args.count}: injecting {args.fault} (index {index})")
        generate_one_flight(conn, args.fault, args.out_dir, args.sitl_log_dir, index)


if __name__ == "__main__":
    main()
