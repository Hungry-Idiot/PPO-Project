"""
Simple vs Simple 双连接诊断 v2 — InitData 后直接发 CtrlData，
服务器可能需要双方动作到达后才开始仿真。
"""
import sys
import numpy as np
import socket
import struct
import yaml

CONFIG_PATH = "./config/envs.yaml"
INITIAL_FMT = "<26i296x"
GET_FMT = "=27d"
SEND_FMT = "<5d"


def get_packet(sock, size):
    data = b''
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Socket closed")
        data += chunk
    return data


def main():
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    my_init = np.array([
        0.0, 0.0, 1000.0, 0.0, 0.0, 0.0,
        5.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ], dtype=np.float64)

    enemy_init = np.array([
        20.0, 0.0, 1000.0, 0.0, 0.0, 0.0,
        5.0, 3.0, 0.0, 0.0, 0.0, 0.0
    ], dtype=np.float64)

    # Connect both
    sock_a = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_a.connect((config['host'], config['port']))
    sock_b = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_b.connect((config['host'], config['port']))
    print("[OK] Both connected")

    # Send InitData for both
    init_a = np.array([114514, 1919810], dtype=np.int32)
    init_a = np.append(init_a, np.append(my_init, enemy_init).astype(np.int32))
    sock_a.send(struct.pack(INITIAL_FMT, *init_a))

    init_b = np.array([114514, 1919810], dtype=np.int32)
    init_b = np.append(init_b, np.append(enemy_init, my_init).astype(np.int32))
    sock_b.send(struct.pack(INITIAL_FMT, *init_b))
    print("[OK] Both InitData sent")

    # CRITICAL: send CtrlData BEFORE trying to recv — server may need
    # both sides' actions before producing any BattleData
    zero_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    sock_a.send(struct.pack(SEND_FMT, *zero_action))
    sock_b.send(struct.pack(SEND_FMT, *zero_action))
    print("[OK] Both CtrlData sent, now waiting for BattleData...")

    prev_enemy_hp = 1.0
    for step in range(30):
        # Read BattleData for both
        raw_a = np.array(struct.unpack(GET_FMT, get_packet(sock_a, 216)), dtype=np.float64)
        raw_b = np.array(struct.unpack(GET_FMT, get_packet(sock_b, 216)), dtype=np.float64)

        dist = np.linalg.norm(raw_a[13:16] - raw_a[0:3]) * 10.0
        dmg = max(0.0, (prev_enemy_hp - raw_a[25]) * 1000.0)
        prev_enemy_hp = raw_a[25]

        print(f"  step {step+1}: dist={dist:.0f}m, my_hp={raw_a[12]:.4f}, enemy_hp={raw_a[25]:.4f}, "
              f"dmg={dmg:.1f}, my_vel={np.linalg.norm(raw_a[6:9]):.1f}")

        # Send next actions
        sock_a.send(struct.pack(SEND_FMT, *zero_action))
        sock_b.send(struct.pack(SEND_FMT, *zero_action))

    sock_a.close()
    sock_b.close()
    print("\n[DONE]")


if __name__ == "__main__":
    main()
