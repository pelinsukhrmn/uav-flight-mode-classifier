# SITL'e bağlanıp canlı arıza tahminini bilinen enjeksiyon programıyla karşılaştıran ve web sayfasında gösteren izleyici.
import argparse
import json
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from pymavlink import mavutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from live_inference import LiveWindowBuffer, score_window
from inference_common import load_artifacts

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sitl_generate_fault_logs import FAULT_PARAMS, FAULT_CLEAR, set_param

STATE_LOCK = threading.Lock()
STATE = {
    "elapsed_s": 0.0,
    "current_label": None, "current_conf": 0.0,
    "next_label": None, "next_conf": 0.0,
    "true_fault": "normal",
    "n_scored": 0, "n_correct_now": 0,
    "history": [],
    "custom_mode": None, "armed": False,
    "last_statustext": None, "last_arm_ack": None, "last_takeoff_ack": None,
    "relative_alt_m": None, "ekf_flags": 0, "landed_state": None, "home_set": False,
}
EKF_POS_HORIZ_ABS = mavutil.mavlink.EKF_POS_HORIZ_ABS
EKF_POS_VERT_ABS = mavutil.mavlink.EKF_POS_VERT_ABS
FAULT_SCHEDULE = []
FLIGHT_START_WALL = None


def heartbeat_loop(conn):
    while True:
        conn.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(1.0)


def true_fault_at(elapsed_s):
    for window in FAULT_SCHEDULE:
        if window["start_s"] <= elapsed_s < window["end_s"]:
            return window["fault"]
    return "normal"


def reader_loop(conn, meta, window_queue, min_interval_s=1.0):
    buffer = LiveWindowBuffer(meta)
    latest_attitude = None
    last_scored_ts = None

    while True:
        msg = conn.recv_match(blocking=True, timeout=1)
        if msg is None:
            continue
        msg_type = msg.get_type()

        if msg_type == "HEARTBEAT":
            armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            with STATE_LOCK:
                STATE["custom_mode"] = msg.custom_mode
                STATE["armed"] = armed
            continue

        if msg_type == "ATTITUDE":
            latest_attitude = msg
            continue

        if msg_type == "STATUSTEXT":
            with STATE_LOCK:
                STATE["last_statustext"] = msg.text
            continue

        if msg_type == "COMMAND_ACK":
            with STATE_LOCK:
                if msg.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
                    STATE["last_arm_ack"] = msg.result
                elif msg.command == mavutil.mavlink.MAV_CMD_NAV_TAKEOFF:
                    STATE["last_takeoff_ack"] = msg.result
            continue

        if msg_type == "EKF_STATUS_REPORT":
            with STATE_LOCK:
                STATE["ekf_flags"] = msg.flags
            continue

        if msg_type == "EXTENDED_SYS_STATE":
            with STATE_LOCK:
                STATE["landed_state"] = msg.landed_state
            continue

        if msg_type == "HOME_POSITION":
            with STATE_LOCK:
                STATE["home_set"] = True
            continue

        if msg_type != "LOCAL_POSITION_NED" or latest_attitude is None:
            continue

        with STATE_LOCK:
            STATE["relative_alt_m"] = -msg.z

        row = {
            "timestamp": msg.time_boot_ms * 1000,
            "vertical_speed": -msg.vz,
            "horizontal_speed": (msg.vx ** 2 + msg.vy ** 2) ** 0.5,
            "roll_angle": np.degrees(latest_attitude.roll),
            "pitch_angle": np.degrees(latest_attitude.pitch),
        }
        buffer.push(row)

        window, end_ts = buffer.latest_window()
        if window is None:
            continue
        if last_scored_ts is not None and (end_ts - last_scored_ts) / 1e6 < min_interval_s:
            continue
        last_scored_ts = end_ts

        elapsed = (time.time() - FLIGHT_START_WALL) if FLIGHT_START_WALL else 0.0
        try:
            window_queue.put_nowait((window, elapsed))
        except queue.Full:
            pass


def scorer_loop(window_queue, scaler, model, next_scaler, next_model, meta, next_meta):
    while True:
        window, elapsed = window_queue.get()

        current_label, current_conf = score_window(window, scaler, model, meta)
        next_label, next_conf = score_window(window, next_scaler, next_model, next_meta)
        true_fault = true_fault_at(elapsed)

        with STATE_LOCK:
            STATE["elapsed_s"] = elapsed
            STATE["current_label"] = current_label
            STATE["current_conf"] = current_conf
            STATE["next_label"] = next_label
            STATE["next_conf"] = next_conf
            STATE["true_fault"] = true_fault
            STATE["n_scored"] += 1
            if current_label == true_fault:
                STATE["n_correct_now"] += 1
            STATE["history"].append({
                "elapsed_s": round(elapsed, 1), "true_fault": true_fault,
                "current_label": current_label, "current_conf": round(current_conf, 2),
                "next_label": next_label, "next_conf": round(next_conf, 2),
            })
            STATE["history"] = STATE["history"][-40:]


def wait_for_mode(conn, mode_name, timeout=30):
    mode_id = conn.mode_mapping()[mode_name]
    conn.set_mode(mode_id)
    deadline = time.time() + timeout
    while time.time() < deadline:
        with STATE_LOCK:
            if STATE["custom_mode"] == mode_id:
                return True
        time.sleep(0.2)
    return False


def wait_for_armed(conn, armed, timeout=45, resend_interval=3):
    deadline = time.time() + timeout
    last_send = 0.0
    seen_ack = None
    seen_statustext = None
    while time.time() < deadline:
        if time.time() - last_send >= resend_interval:
            conn.mav.command_long_send(
                conn.target_system, conn.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1 if armed else 0, 0, 0, 0, 0, 0, 0,
            )
            last_send = time.time()
        with STATE_LOCK:
            if STATE["armed"] == armed:
                return True
            ack = STATE["last_arm_ack"]
            statustext = STATE["last_statustext"]
        if ack is not None and ack != seen_ack:
            print(f"  arm rejected: MAV_RESULT={ack}, retrying", flush=True)
            seen_ack = ack
        if statustext is not None and statustext != seen_statustext:
            print(f"  STATUSTEXT: {statustext}", flush=True)
            seen_statustext = statustext
        time.sleep(0.2)
    return False


def wait_for_ekf_position_ok(conn, timeout=30):
    required = EKF_POS_HORIZ_ABS | EKF_POS_VERT_ABS
    deadline = time.time() + timeout
    while time.time() < deadline:
        with STATE_LOCK:
            flags = STATE["ekf_flags"]
        if flags & required == required:
            return True
        time.sleep(0.2)
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
        with STATE_LOCK:
            if STATE["home_set"]:
                return True
        time.sleep(0.2)
    return False


def send_takeoff_with_retry(conn, altitude, timeout=20, resend_interval=3):
    deadline = time.time() + timeout
    last_send = 0.0
    seen_ack = None
    seen_statustext = None
    while time.time() < deadline:
        if time.time() - last_send >= resend_interval:
            with STATE_LOCK:
                landed_state = STATE["landed_state"]
                alt = STATE["relative_alt_m"]
            print(f"  sending takeoff, landed_state={landed_state} relative_alt_m={alt}", flush=True)
            conn.mav.command_long_send(
                conn.target_system, conn.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, altitude,
            )
            last_send = time.time()
        with STATE_LOCK:
            ack = STATE["last_takeoff_ack"]
            statustext = STATE["last_statustext"]
        if ack is not None and ack != seen_ack:
            print(f"  takeoff ack: MAV_RESULT={ack}", flush=True)
            seen_ack = ack
            if ack == 0:
                return True
        if statustext is not None and statustext != seen_statustext:
            print(f"  STATUSTEXT: {statustext}", flush=True)
            seen_statustext = statustext
        time.sleep(0.2)
    return False


def flight_loop(conn, fault_sequence, background_s, hold_s, recover_s, altitude, climb_wait_s):
    global FLIGHT_START_WALL
    FLIGHT_START_WALL = time.time()

    wait_for_mode(conn, "GUIDED")
    if not wait_for_armed(conn, True):
        print("WARNING: never confirmed armed - proceeding anyway, flight will likely be invalid", flush=True)

    if not wait_for_ekf_position_ok(conn):
        print("WARNING: EKF position never reported ok - attempting takeoff anyway", flush=True)

    if not wait_for_home_position(conn):
        print("WARNING: home position never confirmed - attempting takeoff anyway", flush=True)

    print("Settling 8s after arming before takeoff (re-arming if needed)...", flush=True)
    settle_deadline = time.time() + 8
    while time.time() < settle_deadline:
        with STATE_LOCK:
            armed = STATE["armed"]
        if not armed:
            print("  re-arming during settle wait...", flush=True)
            wait_for_armed(conn, True, timeout=10)
        time.sleep(0.5)

    if not send_takeoff_with_retry(conn, altitude):
        print("WARNING: takeoff command never ACKed as accepted - proceeding anyway", flush=True)

    climb_deadline = time.time() + climb_wait_s
    while time.time() < climb_deadline:
        with STATE_LOCK:
            armed = STATE["armed"]
            alt = STATE["relative_alt_m"]
        if not armed:
            print(f"  WARNING: disarmed during climb wait (alt={alt})", flush=True)
        time.sleep(1.0)

    for fault_type in fault_sequence:
        time.sleep(background_s)
        start_s = time.time() - FLIGHT_START_WALL
        FAULT_SCHEDULE.append({"fault": fault_type, "start_s": start_s, "end_s": start_s + hold_s})
        for name, value in FAULT_PARAMS[fault_type]:
            set_param(conn, name, value)
        time.sleep(hold_s)
        for name, value in FAULT_CLEAR[fault_type]:
            set_param(conn, name, value)
        time.sleep(recover_s)

    wait_for_mode(conn, "RTL")


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Fault precursor monitor</title>
<style>body{font-family:monospace;background:#111;color:#eee;padding:20px}
.ok{color:#4caf50}.bad{color:#f44336}table{border-collapse:collapse}td,th{padding:4px 10px;text-align:left}</style>
</head><body>
<h2>UAV fault precursor - canli izleme</h2>
<div id="summary"></div>
<table id="hist"><thead><tr><th>t(s)</th><th>gercek</th><th>simdi</th><th>~1-2s sonra</th></tr></thead><tbody></tbody></table>
<script>
async function tick(){
  const r = await fetch('/status'); const s = await r.json();
  const acc = s.n_scored ? (100*s.n_correct_now/s.n_scored).toFixed(1) : '0.0';
  document.getElementById('summary').innerHTML =
    `<p>t=${s.elapsed_s.toFixed(1)}s &nbsp; armed: <b>${s.armed}</b> &nbsp; alt: <b>${s.relative_alt_m}</b>m &nbsp; gercek ariza durumu: <b>${s.true_fault}</b></p>`+
    `<p>simdi: <b>${s.current_label}</b> (${s.current_conf.toFixed(2)}) &nbsp; `+
    `~1-2s sonra: <b>${s.next_label}</b> (${s.next_conf.toFixed(2)})</p>`+
    `<p>calisma-suresi dogruluk (nowcast): <b>${acc}%</b> (${s.n_correct_now}/${s.n_scored})</p>`;
  const tbody = document.querySelector('#hist tbody');
  tbody.innerHTML = s.history.slice().reverse().map(h =>
    `<tr class="${h.current_label===h.true_fault?'ok':'bad'}"><td>${h.elapsed_s}</td><td>${h.true_fault}</td>`+
    `<td>${h.current_label} (${h.current_conf})</td><td>${h.next_label} (${h.next_conf})</td></tr>`).join('');
}
setInterval(tick, 1000); tick();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/status":
            with STATE_LOCK:
                body = json.dumps(STATE).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(PAGE.encode())

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="Live SITL fault-precursor monitor with a web dashboard.")
    parser.add_argument("--connection", default="udp:127.0.0.1:14550")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--faults", nargs="+", default=["motor_out", "gps_glitch", "wind_gust_upset", "sensor_freeze"])
    parser.add_argument("--background-s", type=float, default=20.0)
    parser.add_argument("--hold-s", type=float, default=15.0)
    parser.add_argument("--recover-s", type=float, default=15.0)
    parser.add_argument("--altitude", type=float, default=20.0)
    parser.add_argument("--climb-wait-s", type=float, default=15.0)
    args = parser.parse_args()

    meta, scaler, model = load_artifacts("fault")
    next_meta, next_scaler, next_model = load_artifacts("fault_next")

    conn = mavutil.mavlink_connection(args.connection)
    conn.wait_heartbeat()
    conn.mav.request_data_stream_send(conn.target_system, conn.target_component, mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1)
    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
        mavutil.mavlink.MAVLINK_MSG_ID_EXTENDED_SYS_STATE, 500000, 0, 0, 0, 0, 0,
    )

    heartbeat_thread = threading.Thread(target=heartbeat_loop, args=(conn,), daemon=True)
    heartbeat_thread.start()

    window_queue = queue.Queue(maxsize=4)
    reader_thread = threading.Thread(target=reader_loop, args=(conn, meta, window_queue), daemon=True)
    reader_thread.start()

    scorer_thread = threading.Thread(
        target=scorer_loop, args=(window_queue, scaler, model, next_scaler, next_model, meta, next_meta), daemon=True,
    )
    scorer_thread.start()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"Dashboard: http://localhost:{args.port}", flush=True)

    flight_loop(conn, args.faults, args.background_s, args.hold_s, args.recover_s, args.altitude, args.climb_wait_s)

    print("Flight sequence complete - dashboard stays up, Ctrl+C to stop.", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
