"""
Weapon range measurement - single connection version.
One TCP connection for all tests. No reconnect pollution.
"""
import sys
sys.path.insert(0, './envs')
sys.path.insert(0, './utils')
import numpy as np
import struct
import socket
import yaml

CONFIG_PATH = './config/envs.yaml'
INITIAL_FMT = "<26i296x"
GET_FMT = "=27d"
SEND_FMT = "<5d"

def get_packet(sock, size):
    data = b''
    while len(data) < size:
        data += sock.recv(size - len(data))
    return data

def send_packet(sock, fmt, data):
    sock.send(struct.pack(fmt, *data))

def send_init(sock, dist_x):
    my_init = np.array([
        0.0, 0.0, 1000.0,
        0.0, 0.0, 0.0,
        0.5, 0.0, 0.0,
        0.0, 0.0, 0.0
    ], dtype=np.float64)
    enemy_init = np.array([
        dist_x, 0.0, 1000.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ], dtype=np.float64)
    init_state = np.append(my_init, enemy_init)
    init_packet = np.array([114514, 1919810], dtype=np.int32)
    init_packet = np.append(init_packet, init_state.astype(np.int32))
    send_packet(sock, INITIAL_FMT, init_packet)

def read_first_obs(sock):
    raw = np.array(struct.unpack(GET_FMT, get_packet(sock, 216)), dtype=np.float64)
    return raw

def step_zero(sock):
    raw = read_first_obs(sock)
    hp_before = raw[25]
    my_pos_after_init = raw[0:3].copy()
    enemy_pos_after_init = raw[13:16].copy()

    zero_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    send_packet(sock, SEND_FMT, zero_action)
    raw2 = np.array(struct.unpack(GET_FMT, get_packet(sock, 216)), dtype=np.float64)
    hp_after = raw2[25]
    my_pos_after_step = raw2[0:3].copy()

    damage = hp_before - hp_after
    actual_dist = np.linalg.norm(enemy_pos_after_init - my_pos_after_init)
    return damage, actual_dist, my_pos_after_init, enemy_pos_after_init, my_pos_after_step


def main():
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("Weapon Range Measurement - Single Connection")
    print("One TCP connection, no reconnect pollution")
    print("=" * 60)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((config['host'], config['port']))

    # Verify basic connectivity with known-working distance
    print("\n[Verify] Baseline test: init_dist=3 (30m)")
    send_init(sock, 3.0)
    damage, actual_dist, my_pos, enemy_pos, my_pos2 = step_zero(sock)
    print(f"  init_dist=3: my_pos={my_pos}, enemy_pos={enemy_pos}, actual_dist={actual_dist:.1f}({actual_dist*10:.0f}m)")
    print(f"  damage: {damage*1000:.1f} HP  {'[OK] baseline works' if damage > 1e-12 else '[FAIL] connection broken!'}")

    if damage < 1e-12:
        print("\n[ERROR] No damage at 30m! Room may need recreation. Abort.")
        sock.close()
        return

    # Phase 1: Scan range 2..50 with step 2
    print("\n[Phase 1] Scan: dist 2..50, step 2")
    for dist_x in range(2, 52, 2):
        send_init(sock, float(dist_x))
        damage, actual_dist, my_pos, enemy_pos, _ = step_zero(sock)
        dmg_hp = damage * 1000.0
        triggered = dmg_hp > 0.5
        marker = "DMG" if triggered else "---"
        print(f"  init={dist_x:3d}({dist_x*10:4d}m) actual={actual_dist:.1f}({actual_dist*10:.0f}m) "
              f"{marker} dmg={dmg_hp:.1f}")

    # Phase 2: Binary search for exact boundary
    print("\n[Phase 2] Binary search for range boundary")
    # Rough scan to find boundary
    lo, hi = 2.0, 50.0
    for dist_x in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        send_init(sock, float(dist_x))
        damage, _, _, _, _ = step_zero(sock)
        triggered = damage * 1000.0 > 0.5
        if triggered:
            lo = float(dist_x)
        else:
            hi = float(dist_x)
            break

    print(f"  Rough boundary: {lo}({lo*10:.0f}m) ~ {hi}({hi*10:.0f}m)")

    for _ in range(8):
        mid = (lo + hi) / 2.0
        send_init(sock, mid)
        damage, actual_dist, _, _, _ = step_zero(sock)
        triggered = damage * 1000.0 > 0.5
        print(f"  [{lo:.2f}({lo*10:.0f}m) , {hi:.2f}({hi*10:.0f}m)] mid={mid:.3f}({mid*10:.1f}m) actual={actual_dist:.1f}: "
              f"{'DMG' if triggered else '---'} dmg={damage*1000:.1f}")
        if triggered:
            lo = mid
        else:
            hi = mid

    print(f"\n  >> Weapon range boundary: {lo:.2f} ~ {hi:.2f} units = {lo*10:.0f}m ~ {hi*10:.0f}m")

    # Phase 3: Angle test
    print("\n[Phase 3] Angle test: dist=3, vary yaw")
    for yaw_deg in [0, 20, 45, 90, 135, 180]:
        yaw_rad = np.radians(yaw_deg)
        my_init = np.array([0.0, 0.0, 1000.0, 0.0, 0.0, yaw_rad, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        enemy_init = np.array([3.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        init_state = np.append(my_init, enemy_init)
        init_packet = np.array([114514, 1919810], dtype=np.int32)
        init_packet = np.append(init_packet, init_state.astype(np.int32))
        send_packet(sock, INITIAL_FMT, init_packet)
        raw = read_first_obs(sock)
        hp_before = raw[25]
        zero_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        send_packet(sock, SEND_FMT, zero_action)
        raw2 = np.array(struct.unpack(GET_FMT, get_packet(sock, 216)), dtype=np.float64)
        hp_after = raw2[25]
        damage = (hp_before - hp_after) * 1000.0
        marker = "DMG" if damage > 0.5 else "---"
        print(f"  dist=3, yaw={yaw_deg:3d}deg -> {marker}  dmg={damage:.1f}")

    # Phase 4: Reproduce training scenario
    print("\n[Phase 4] Training scenario: init_dist=20, vel=5, zero action x50 steps")
    my_init = np.array([
        0.0, 0.0, 1000.0,
        0.0, 0.0, 0.0,
        5.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ], dtype=np.float64)
    enemy_init = np.array([
        20.0, 0.0, 1000.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ], dtype=np.float64)
    init_state = np.append(my_init, enemy_init)
    init_packet = np.array([114514, 1919810], dtype=np.int32)
    init_packet = np.append(init_packet, init_state.astype(np.int32))
    send_packet(sock, INITIAL_FMT, init_packet)
    raw = read_first_obs(sock)
    print(f"  init: my_pos={raw[0:3]}, enemy_pos={raw[13:16]}, "
          f"dist={np.linalg.norm(raw[13:16]-raw[0:3]):.1f}({np.linalg.norm(raw[13:16]-raw[0:3])*10:.0f}m)")
    prev_hp = raw[25]
    total_dmg = 0.0
    for step in range(50):
        zero_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        send_packet(sock, SEND_FMT, zero_action)
        raw = np.array(struct.unpack(GET_FMT, get_packet(sock, 216)), dtype=np.float64)
        hp_change = (prev_hp - raw[25]) * 1000.0
        total_dmg += max(0, hp_change)
        if hp_change > 0.5:
            dist = np.linalg.norm(raw[13:16] - raw[0:3])
            print(f"    step {step+1:2d}: dist={dist:.1f}({dist*10:.0f}m) dmg={hp_change:.1f} "
                  f"enemy_hp={raw[25]:.10f}")
        prev_hp = raw[25]

    print(f"  Total damage: {total_dmg:.1f} HP")
    print(f"  {'[OK] Weapon sustained!' if total_dmg > 10 else '[WARN] Weapon not sustained!'}")

    sock.close()
    print(f"\n{'=' * 60}")
    print("Measurement complete (single connection, reliable)")
    print("=" * 60)


if __name__ == "__main__":
    main()
