"""
近距离交战诊断：测试agent接近敌机时是否触发伤害
"""
import sys
sys.path.insert(0, './envs')
sys.path.insert(0, './utils')
import numpy as np
from utils import adaptor, action, observation, reward, initialize
import yaml

CONFIG = './config/envs.yaml'

def main():
    net = adaptor.NetworkAdaptor(CONFIG)
    net.connect()
    init_state = initialize.generate_initial_state()
    init_packet = np.array([114514, 1919810], dtype=np.int32)
    init_packet = np.append(init_packet, init_state.astype(np.int32))
    net.send_initial_packet(init_packet)

    raw = net.get_observation_packet()
    my_s = raw[0:13].astype(np.float64)
    enemy_s = raw[13:26].astype(np.float64)

    enemy_start_hp = enemy_s[12]
    print(f"初始: 己方pos={my_s[0:3]} hp={my_s[12]:.6f}, 敌方pos={enemy_s[0:3]} hp={enemy_s[12]:.6f}")

    min_dist = float('inf')
    total_damage = 0.0
    total_steps = 200

    print(f"\n{'步':>4s} {'距离':>8s} {'敌HP':>10s} {'HP变化':>10s} {'伤害奖励':>10s} {'追近':>8s} {'朝向':>8s}")
    print("-" * 70)

    for step in range(total_steps):
        prev_my = my_s.copy()
        prev_enemy_s = enemy_s.copy()

        rel = enemy_s[0:3] - my_s[0:3]
        dist = np.linalg.norm(rel)
        min_dist = min(min_dist, dist)

        target_dir = rel / (dist + 1e-6)
        desired_yaw = np.arctan2(target_dir[1], target_dir[0])
        desired_pitch = np.arcsin(np.clip(target_dir[2], -1, 1))
        current_yaw = my_s[5]
        current_pitch = my_s[4]
        yaw_err = desired_yaw - current_yaw
        while yaw_err > np.pi: yaw_err -= 2*np.pi
        while yaw_err < -np.pi: yaw_err += 2*np.pi
        pitch_err = desired_pitch - current_pitch

        test_action = np.array([1.0, np.clip(pitch_err * 2.0, -1, 1), 0.0, np.clip(yaw_err * 2.0, -1, 1)], dtype=np.float64)

        real_action = action.marshal_action(test_action)
        send_pack = np.append(real_action, 0.0)
        net.send_action_packet(send_pack)

        raw = net.get_observation_packet()
        my_s = raw[0:13].astype(np.float64)
        enemy_s = raw[13:26].astype(np.float64)
        is_done = raw[26]

        comps = reward.reward_components(prev_my, prev_enemy_s, my_s, enemy_s)
        hp_delta = (prev_enemy_s[12] - enemy_s[12]) * 1000.0
        total_damage += max(0, hp_delta)

        print(f"{step+1:4d} {dist:8.1f} {enemy_s[12]:10.6f} {hp_delta:10.4f} {comps['damage_reward']:10.4f} "
              f"{comps['approach_reward']:8.3f} {comps['heading_reward']:8.3f}")

        if is_done:
            print(f"\n[提前终止] 第{step+1}步 is_done=True")
            break

    print(f"\n{'='*60}")
    print(f"诊断结果:")
    print(f"  最小距离: {min_dist:.1f} (10m单位 = {min_dist*10:.0f}m)")
    print(f"  总伤害量: {total_damage:.4f}")
    print(f"  最终己方HP: {my_s[12]:.6f}")
    print(f"  最终敌方HP: {enemy_s[12]:.6f}")
    print(f"  敌方HP变化: {(enemy_start_hp - enemy_s[12])*1000:.4f}")
    if total_damage == 0.0:
        print(f"\n  *** 诊断结论: 200步内无任何伤害 ***")
        print(f"  可能原因:")
        print(f"  1. 仿真平台武器不自瞄，需要额外开火指令")
        print(f"  2. 我方武器射程不足（最近距离={min_dist*10:.0f}m）")
        print(f"  3. 固定靶机对特定武器类型免疫")
    print(f"{'='*60}")

    net.socket.close()

if __name__ == "__main__":
    main()
