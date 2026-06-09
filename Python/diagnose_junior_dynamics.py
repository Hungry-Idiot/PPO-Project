from __future__ import annotations

import argparse
import csv
import socket
import struct
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


CONFIG_PATH = "./config/envs_junior.yaml"

INITIAL_FMT = "<26i296x"
GET_FMT = "=27d"
SEND_FMT = "<5d"

INITIAL_PACKET_SIZE = 400
GET_PACKET_SIZE = 216
SEND_PACKET_SIZE = 40


def get_nested(config, keys, default=None):
    cur = config
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def connect_socket(host, port, timeout_seconds):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout_seconds)
    print(f"[CONNECT] {host}:{port}", flush=True)
    sock.connect((host, port))
    print(f"[OK] connected to {host}:{port}", flush=True)
    return sock


def recv_exact(sock, packet_size):
    data = b""

    while len(data) < packet_size:
        try:
            chunk = sock.recv(packet_size - len(data))
        except socket.timeout:
            raise TimeoutError(
                f"Timeout waiting for packet: got {len(data)}/{packet_size} bytes"
            )

        if not chunk:
            raise ConnectionError(
                f"Socket closed while waiting for packet: got {len(data)}/{packet_size} bytes"
            )

        data += chunk

    return data


def send_initial_packet(sock, initial_packet):
    packed = struct.pack(INITIAL_FMT, *initial_packet)

    if len(packed) != INITIAL_PACKET_SIZE:
        raise RuntimeError(f"Bad InitData size: {len(packed)}")

    sock.sendall(packed)


def send_ctrl_packet(sock, ctrl_packet):
    packed = struct.pack(SEND_FMT, *ctrl_packet)

    if len(packed) != SEND_PACKET_SIZE:
        raise RuntimeError(f"Bad CtrlData size: {len(packed)}")

    sock.sendall(packed)


def recv_battle_packet(sock):
    raw = recv_exact(sock, GET_PACKET_SIZE)
    return np.array(struct.unpack(GET_FMT, raw), dtype=np.float64)


def split_battle_packet(raw):
    my_state = raw[0:13].astype(np.float64).copy()
    enemy_state = raw[13:26].astype(np.float64).copy()
    is_done = bool(raw[26] == 1.0)
    return my_state, enemy_state, is_done


def build_state(config, side):
    section = get_nested(config, ["initial_state", side], {})

    pos = section.get("pos", [0.0, 0.0, 1000.0])
    rot = section.get("rot", [0.0, 0.0, 0.0])
    vel = section.get("vel", [10.0, 0.0, 0.0])
    ang_vel = section.get("ang_vel", [0.0, 0.0, 0.0])

    state = np.array(pos + rot + vel + ang_vel, dtype=np.float64)

    if state.shape != (12,):
        raise ValueError(f"Initial state for {side} must have 12 values, got {state.shape}")

    return state


def pack_initial(my_init, enemy_init, room_id, unit_id):
    packet = np.array([room_id, unit_id], dtype=np.int32)
    packet = np.append(packet, my_init.astype(np.int32))
    packet = np.append(packet, enemy_init.astype(np.int32))

    if packet.shape != (26,):
        raise ValueError(f"InitData must contain 26 int32 values, got {packet.shape}")

    return packet


def pack_ctrl(throttle, pitch, roll, yaw, done=0.0):
    throttle = float(np.clip(throttle, 0.0, 1.0))
    pitch = float(np.clip(pitch, -1.0, 1.0))
    roll = float(np.clip(roll, -1.0, 1.0))
    yaw = float(np.clip(yaw, -1.0, 1.0))
    done = float(done)

    return np.array([throttle, pitch, roll, yaw, done], dtype=np.float64)


def load_test_cases(config):
    cases = get_nested(config, ["test_cases"], None)

    if not cases:
        max_steps = int(get_nested(config, ["diagnostics", "max_steps_per_case"], 300))
        cases = [
            {
                "name": "zero_action",
                "throttle": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
                "yaw": 0.0,
                "steps": 180,
            },
            {
                "name": "throttle_0_4",
                "throttle": 0.4,
                "pitch": 0.0,
                "roll": 0.0,
                "yaw": 0.0,
                "steps": max_steps,
            },
            {
                "name": "throttle_0_6",
                "throttle": 0.6,
                "pitch": 0.0,
                "roll": 0.0,
                "yaw": 0.0,
                "steps": max_steps,
            },
            {
                "name": "throttle_0_8",
                "throttle": 0.8,
                "pitch": 0.0,
                "roll": 0.0,
                "yaw": 0.0,
                "steps": max_steps,
            },
        ]

    normalized = []
    default_steps = int(get_nested(config, ["diagnostics", "max_steps_per_case"], 300))

    for idx, case in enumerate(cases):
        normalized.append(
            {
                "name": str(case.get("name", f"case_{idx}")),
                "throttle": float(case.get("throttle", 0.0)),
                "pitch": float(case.get("pitch", 0.0)),
                "roll": float(case.get("roll", 0.0)),
                "yaw": float(case.get("yaw", 0.0)),
                "steps": int(case.get("steps", default_steps)),
            }
        )

    return normalized


def select_cases(cases, case_name):
    if case_name is None:
        return cases

    if case_name == "all":
        return cases

    selected = [case for case in cases if case["name"] == case_name]

    if not selected:
        names = [case["name"] for case in cases]
        raise ValueError(
            "Unknown case: "
            + case_name
            + "\nAvailable cases:\n  "
            + "\n  ".join(names)
        )

    return selected


def safe_close(sock):
    if sock is None:
        return

    try:
        sock.close()
    except Exception:
        pass

def request_episode_end(my_sock, enemy_sock=None, dual_port=False):
    """
    主动请求仿真平台结束当前 episode。
    CtrlData 第 5 个字段 is_done=1.0。
    """
    done_ctrl = pack_ctrl(
        throttle=0.0,
        pitch=0.0,
        roll=0.0,
        yaw=0.0,
        done=1.0,
    )

    sockets = [my_sock]

    if dual_port and enemy_sock is not None:
        sockets.append(enemy_sock)

    for sock in sockets:
        if sock is None:
            continue

        try:
            send_ctrl_packet(sock, done_ctrl)
        except Exception as exc:
            print(f"[WARN] failed to send episode-end CtrlData: {repr(exc)}", flush=True)

    # 有些平台会在收到 is_done 后返回最后一个 BattleData；
    # 有些可能直接关闭/不返回。这里尝试接收，但失败不影响后续。
    for sock in sockets:
        if sock is None:
            continue

        try:
            _ = recv_battle_packet(sock)
        except Exception:
            pass

def run_one_case(config, case, args):
    host = args.host if args.host is not None else str(config.get("host", "127.0.0.1"))
    port = args.port if args.port is not None else int(config.get("port", 1000))

    timeout_seconds = float(
        args.timeout
        if args.timeout is not None
        else get_nested(config, ["diagnostics", "timeout_seconds"], 10.0)
    )

    print_every = int(
        args.print_every
        if args.print_every is not None
        else get_nested(config, ["diagnostics", "print_every"], 10)
    )

    config_dual_port = bool(get_nested(config, ["room", "dual_port"], False))
    dual_port = args.dual_port if args.dual_port is not None else config_dual_port

    enemy_port_offset = int(get_nested(config, ["room", "enemy_port_offset"], 1))
    enemy_port = port + enemy_port_offset

    room_id = int(get_nested(config, ["room", "room_id"], 114514))
    my_unit_id = int(get_nested(config, ["room", "my_unit_id"], 1919810))
    enemy_unit_id = int(get_nested(config, ["room", "enemy_unit_id"], 1919811))

    my_init = build_state(config, "my")
    enemy_init = build_state(config, "enemy")

    expected_initial_z = float(my_init[2])
    expected_initial_speed = float(np.linalg.norm(my_init[6:9]))

    my_sock = None
    enemy_sock = None
    episode_started = False

    rows = []
    summary = {
        "case": case["name"],
        "ok": 0,
        "error": "",
        "steps_run": 0,
        "done": 0,
        "self_dead": 0,
        "start_z_units": np.nan,
        "start_speed_units": np.nan,
        "start_dist_meters": np.nan,
        "final_z_units": np.nan,
        "delta_z_units": np.nan,
        "min_z_units": np.nan,
        "max_z_units": np.nan,
        "final_speed_units": np.nan,
        "min_speed_units": np.nan,
        "max_speed_units": np.nan,
        "final_hp": np.nan,
        "final_enemy_hp": np.nan,
        "total_damage_taken_hp": 0.0,
        "total_damage_dealt_hp": 0.0,
        "initial_state_warning": "",
    }

    try:
        my_sock = connect_socket(host, port, timeout_seconds)

        if dual_port:
            enemy_sock = connect_socket(host, enemy_port, timeout_seconds)

        packet_my = pack_initial(my_init, enemy_init, room_id, my_unit_id)
        send_initial_packet(my_sock, packet_my)

        if dual_port:
            packet_enemy = pack_initial(enemy_init, my_init, room_id, enemy_unit_id)
            send_initial_packet(enemy_sock, packet_enemy)

        ctrl_my = pack_ctrl(
            throttle=case["throttle"],
            pitch=case["pitch"],
            roll=case["roll"],
            yaw=case["yaw"],
            done=0.0,
        )

        ctrl_enemy = pack_ctrl(
            throttle=0.0,
            pitch=0.0,
            roll=0.0,
            yaw=0.0,
            done=0.0,
        )

        # 关键：发送 CtrlData 后才等待 BattleData
        send_ctrl_packet(my_sock, ctrl_my)

        if dual_port:
            send_ctrl_packet(enemy_sock, ctrl_enemy)

        episode_started = True

        initial_z = None
        min_z = float("inf")
        max_z = -float("inf")
        min_speed = float("inf")
        max_speed = 0.0

        prev_my_hp = None
        prev_enemy_hp = None
        total_damage_taken = 0.0
        total_damage_dealt = 0.0

        print(
            f"\n[CASE] {case['name']} | "
            f"throttle={case['throttle']:.2f}, "
            f"pitch={case['pitch']:.2f}, "
            f"roll={case['roll']:.2f}, "
            f"yaw={case['yaw']:.2f}, "
            f"steps={case['steps']}, "
            f"dual_port={dual_port}",
            flush=True,
        )

        print(
            "[EXPECTED INITIAL] "
            f"z={expected_initial_z:.2f}, speed={expected_initial_speed:.2f}",
            flush=True,
        )

        header = (
            f"{'step':>5s} {'z(unit)':>10s} {'z(m)':>10s} "
            f"{'dz(unit)':>10s} {'speed':>9s} {'w':>9s} "
            f"{'dist(m)':>10s} {'my_hp':>8s} {'enemy_hp':>9s} {'done':>6s}"
        )
        print(header)
        print("-" * len(header))

        for step in range(1, int(case["steps"]) + 1):
            raw_my = recv_battle_packet(my_sock)

            if dual_port:
                _ = recv_battle_packet(enemy_sock)

            my_state, enemy_state, is_done = split_battle_packet(raw_my)

            z_units = float(my_state[2])
            z_meters = z_units * 10.0
            speed_units = float(np.linalg.norm(my_state[6:9]))
            vertical_speed_w = float(my_state[8])
            dist_meters = float(np.linalg.norm(enemy_state[0:3] - my_state[0:3]) * 10.0)
            my_hp = float(my_state[12])
            enemy_hp = float(enemy_state[12])

            if initial_z is None:
                initial_z = z_units

                summary["start_z_units"] = z_units
                summary["start_speed_units"] = speed_units
                summary["start_dist_meters"] = dist_meters

                z_diff = abs(z_units - expected_initial_z)
                speed_diff = abs(speed_units - expected_initial_speed)

                warnings = []

                if z_diff > 5.0:
                    warnings.append(
                        f"start_z differs from configured z by {z_diff:.2f} units"
                    )

                if speed_diff > 3.0:
                    warnings.append(
                        f"start_speed differs from configured speed by {speed_diff:.2f} units"
                    )

                if warnings:
                    summary["initial_state_warning"] = "; ".join(warnings)
                    print("[WARN] " + summary["initial_state_warning"], flush=True)

            delta_z = z_units - initial_z
            min_z = min(min_z, z_units)
            max_z = max(max_z, z_units)
            min_speed = min(min_speed, speed_units)
            max_speed = max(max_speed, speed_units)

            damage_taken = 0.0
            damage_dealt = 0.0

            if prev_my_hp is not None:
                damage_taken = max(0.0, (prev_my_hp - my_hp) * 1000.0)

            if prev_enemy_hp is not None:
                damage_dealt = max(0.0, (prev_enemy_hp - enemy_hp) * 1000.0)

            total_damage_taken += damage_taken
            total_damage_dealt += damage_dealt

            prev_my_hp = my_hp
            prev_enemy_hp = enemy_hp

            row = {
                "case": case["name"],
                "step": step,
                "throttle": case["throttle"],
                "pitch": case["pitch"],
                "roll": case["roll"],
                "yaw": case["yaw"],
                "dual_port": int(dual_port),
                "z_units": z_units,
                "z_meters": z_meters,
                "delta_z_units": delta_z,
                "delta_z_meters": delta_z * 10.0,
                "speed_units": speed_units,
                "vertical_speed_w": vertical_speed_w,
                "dist_meters": dist_meters,
                "my_hp": my_hp,
                "enemy_hp": enemy_hp,
                "damage_taken_hp": damage_taken,
                "damage_dealt_hp": damage_dealt,
                "total_damage_taken_hp": total_damage_taken,
                "total_damage_dealt_hp": total_damage_dealt,
                "is_done": int(is_done),
            }
            rows.append(row)

            should_print = (
                step == 1
                or step == int(case["steps"])
                or is_done
                or step % max(1, print_every) == 0
            )

            if should_print:
                print(
                    f"{step:5d} {z_units:10.2f} {z_meters:10.1f} "
                    f"{delta_z:10.2f} {speed_units:9.2f} {vertical_speed_w:9.2f} "
                    f"{dist_meters:10.1f} {my_hp:8.4f} {enemy_hp:9.4f} {str(is_done):>6s}",
                    flush=True,
                )

            summary["steps_run"] = step
            summary["done"] = int(is_done)
            summary["self_dead"] = int(my_hp <= 0.01)
            summary["final_z_units"] = z_units
            summary["delta_z_units"] = delta_z
            summary["min_z_units"] = min_z
            summary["max_z_units"] = max_z
            summary["final_speed_units"] = speed_units
            summary["min_speed_units"] = min_speed
            summary["max_speed_units"] = max_speed
            summary["final_hp"] = my_hp
            summary["final_enemy_hp"] = enemy_hp
            summary["total_damage_taken_hp"] = total_damage_taken
            summary["total_damage_dealt_hp"] = total_damage_dealt

            if is_done or my_hp <= 0.01:
                break

            send_ctrl_packet(my_sock, ctrl_my)

            if dual_port:
                send_ctrl_packet(enemy_sock, ctrl_enemy)

        summary["ok"] = 1

    except Exception as exc:
        summary["ok"] = 0
        summary["error"] = repr(exc)
        print(f"[ERROR] case={case['name']} error={repr(exc)}", flush=True)

    finally:
        if episode_started:
            request_episode_end(
                my_sock=my_sock,
                enemy_sock=enemy_sock,
                dual_port=dual_port,
            )

        safe_close(my_sock)
        safe_close(enemy_sock)
        time.sleep(0.2)

    return rows, summary


def write_csv(path, rows):
    if not rows:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    keys = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def print_case_list(cases):
    print("[CASES]")
    for case in cases:
        print(
            f"  {case['name']}: "
            f"throttle={case['throttle']}, "
            f"pitch={case['pitch']}, "
            f"roll={case['roll']}, "
            f"yaw={case['yaw']}, "
            f"steps={case['steps']}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=CONFIG_PATH)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--print-every", type=int, default=None)
    parser.add_argument("--csv", type=str, default=None)

    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help=(
            "Run only one case by name, for example throttle_0_6. "
            "Use --case all to run all cases in one room, but this is not recommended for clean dynamics diagnosis."
        ),
    )
    parser.add_argument("--list-cases", action="store_true")

    dual_group = parser.add_mutually_exclusive_group()
    dual_group.add_argument("--dual-port", dest="dual_port", action="store_true")
    dual_group.add_argument("--single-port", dest="dual_port", action="store_false")
    parser.set_defaults(dual_port=None)

    args = parser.parse_args()

    config = load_config(args.config)
    all_cases = load_test_cases(config)

    if args.list_cases:
        print_case_list(all_cases)
        return

    selected_cases = select_cases(all_cases, args.case)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.csv:
        csv_path = args.csv
    else:
        if len(selected_cases) == 1:
            case_name_for_path = selected_cases[0]["name"]
        else:
            case_name_for_path = "all"

        csv_template = get_nested(
            config,
            ["diagnostics", "csv_path"],
            "./output/junior_diagnostics/junior_dynamics_{case}_{timestamp}.csv",
        )
        csv_path = csv_template.format(case=case_name_for_path, timestamp=timestamp)

    print("[INFO] Junior dynamics diagnosis")
    print(f"[INFO] Config: {args.config}")
    print(f"[INFO] Selected cases: {len(selected_cases)}")
    print(f"[INFO] CSV: {csv_path}")

    if args.case is None or args.case == "all":
        print()
        print("[INFO] Running multiple cases in one room.")
        print("[INFO] The script will send is_done=1.0 after each case to end the current episode.")
        print(f"[INFO] Make sure maxEpisodes is at least {len(selected_cases) + 2}.")
        print()

    all_rows = []
    summaries = []

    for case in selected_cases:
        rows, summary = run_one_case(config, case, args)
        all_rows.extend(rows)
        summaries.append(summary)

        write_csv(csv_path, all_rows)

    summary_path = str(Path(csv_path).with_name(Path(csv_path).stem + "_summary.csv"))
    write_csv(summary_path, summaries)

    print("\n[SUMMARY]")
    header = (
        f"{'case':>36s} {'ok':>3s} {'steps':>6s} {'done':>5s} "
        f"{'dead':>5s} {'start_z':>9s} {'dz':>9s} {'min_z':>9s} "
        f"{'max_z':>9s} {'final_spd':>10s} {'hp':>8s}"
    )
    print(header)
    print("-" * len(header))

    for item in summaries:
        print(
            f"{item['case']:>36s} {item['ok']:3d} {item['steps_run']:6d} "
            f"{item['done']:5d} {item['self_dead']:5d} "
            f"{item['start_z_units']:9.2f} {item['delta_z_units']:9.2f} "
            f"{item['min_z_units']:9.2f} {item['max_z_units']:9.2f} "
            f"{item['final_speed_units']:10.2f} {item['final_hp']:8.4f}"
        )

        if item["initial_state_warning"]:
            print(f"  warning: {item['initial_state_warning']}")

        if item["error"]:
            print(f"  error: {item['error']}")

    print("\n[DONE]")
    print(f"Detail CSV:  {csv_path}")
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()