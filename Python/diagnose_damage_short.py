"""
最小化伤害触发测试：超近距离 + 极慢速度 + 不动操作
仅验证"攻击范围与受击体积重叠→固定伤害"是否生效
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

def main():
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((config['host'], config['port']))

    # ===== 超近距离初始化 =====
    # 己方: (0, 0, 1000), heading 朝向 +x (yaw=0)
    # 敌方: (3, 0, 1000) — 仅 3 单位 = 30m
    # 己方线速度 u=1.0 (极慢, ~10m/s)
    my_init = np.array([
        0.0, 0.0, 1000.0,   # 位置
        0.0, 0.0, 0.0,      # 姿态 (yaw=0, 机头朝 +x)
        1.0, 0.0, 0.0,      # 线速度 (极慢)
        0.0, 0.0, 0.0       # 角速度
    ], dtype=np.float64)
    enemy_init = np.array([
        3.0, 0.0, 1000.0,   # 位置 (前方 30m)
        0.0, 0.0, 0.0,      # 姿态
        0.0, 0.0, 0.0,      # 速度 (固定靶)
        0.0, 0.0, 0.0
    ], dtype=np.float64)

    init_state = np.append(my_init, enemy_init)
    init_packet = np.array([114514, 1919810], dtype=np.int32)
    init_packet = np.append(init_packet, init_state.astype(np.int32))
    send_packet(sock, INITIAL_FMT, init_packet)

    # 获取初始观测
    raw = np.array(struct.unpack(GET_FMT, get_packet(sock, 216)), dtype=np.float64)
    my_s = raw[0:13].astype(np.float64)
    enemy_s = raw[13:26].astype(np.float64)
    is_done = raw[26]

    print(f"初始己方: pos={my_s[0:3]}, rot={my_s[3:6]}, vel={my_s[6:9]}, hp={my_s[12]:.10f}")
    print(f"初始敌方: pos={enemy_s[0:3]}, rot={enemy_s[3:6]}, vel={enemy_s[6:9]}, hp={enemy_s[12]:.10f}")
    print(f"初始距离: {np.linalg.norm(enemy_s[0:3] - my_s[0:3]):.1f} (10m) = {np.linalg.norm(enemy_s[0:3] - my_s[0:3])*10:.0f}m")
    print(f"is_done: {is_done}")

    prev_hp = enemy_s[12]
    zero_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    print(f"\n{'步':>4s} {'距离':>8s} {'敌HP':>18s} {'HP变化':>14s} {'己方pos':>24s} {'己方rot(yaw)':>14s}")
    print("-" * 90)

    min_dist = float('inf')
    max_hp_change = 0.0

    for step in range(300):
        send_packet(sock, SEND_FMT, zero_action)
        raw = np.array(struct.unpack(GET_FMT, get_packet(sock, 216)), dtype=np.float64)
        my_s = raw[0:13].astype(np.float64)
        enemy_s = raw[13:26].astype(np.float64)
        is_done = raw[26]

        dist = np.linalg.norm(enemy_s[0:3] - my_s[0:3])
        min_dist = min(min_dist, dist)
        hp_change = (prev_hp - enemy_s[12])
        max_hp_change = max(max_hp_change, abs(hp_change))

        if step < 30 or step % 20 == 0 or abs(hp_change) > 1e-12:
            print(f"{step+1:4d} {dist:8.1f} {enemy_s[12]:18.15f} {hp_change:14.12f} "
                  f"{str(my_s[0:3]):>24s} {my_s[5]:14.6f}")

        if abs(hp_change) > 1e-12:
            print(f"  >>> 伤害触发! HP变化={hp_change:.12f} <<<")

        prev_hp = enemy_s[12]

        if is_done:
            print(f"\n[终止] 第{step+1}步 is_done=True")
            break

    print(f"\n{'='*60}")
    print(f"结果:")
    print(f"  总步数: {step+1}")
    print(f"  最小距离: {min_dist:.1f} (10m) = {min_dist*10:.0f}m")
    print(f"  最大HP变化: {max_hp_change:.15f} {'<<< 伤害触发!' if max_hp_change > 1e-12 else '<<< 无任何伤害'}")
    print(f"  最终己方pos: {my_s[0:3]}")
    print(f"  最终敌方pos: {enemy_s[0:3]}")
    print(f"  最终己方rot: {my_s[3:6]}")
    print(f"  最终己方HP: {my_s[12]:.10f}")
    print(f"  最终敌方HP: {enemy_s[12]:.10f}")
    print(f"{'='*60}")

    sock.close()

if __name__ == "__main__":
    main()
