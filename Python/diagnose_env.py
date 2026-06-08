"""
环境连通性诊断脚本
验证TCP通信正常、观测值合理、奖励分量正确
"""
import sys
sys.path.insert(0, './envs')
sys.path.insert(0, './utils')

import numpy as np
from utils import adaptor, action, observation, reward, truncate, initialize
import yaml

CONFIG_PATH = './config/envs.yaml'

def main():
    print("=" * 60)
    print("Phase 1: 环境连通性诊断")
    print("=" * 60)

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    print(f"\n[配置] host={config['host']}, port={config['port']}")

    print("\n[步骤1] 连接TCP服务器...")
    net = adaptor.NetworkAdaptor(CONFIG_PATH)
    net.connect()
    print("  TCP连接成功")

    print("\n[步骤2] 发送初始状态包...")
    init_state = initialize.generate_initial_state()
    init_packet = np.array([114514, 1919810], dtype=np.int32)
    init_packet = np.append(init_packet, init_state.astype(np.int32))
    net.send_initial_packet(init_packet)
    print(f"  己方初始: pos={init_state[0:3]}, vel={init_state[6:9]}")
    print(f"  敌方初始: pos={init_state[12:15]}, vel={init_state[18:21]}")

    print("\n[步骤3] 获取初始观测...")
    try:
        raw_obs = net.get_observation_packet()
    except ConnectionResetError:
        print("  连接被服务器重置! 可能原因:")
        print("  1. 房间已结束 (maxEpisodes用完)")
        print("  2. 房间超时关闭")
        print("  请重新在UE5客户端创建房间，确认端口号后重试")
        return
    my_state = raw_obs[0:13].astype(np.float64)
    enemy_state = raw_obs[13:26].astype(np.float64)
    is_done = raw_obs[26]

    print(f"  己方: pos={my_state[0:3]}, rot={my_state[3:6]}, vel={my_state[6:9]}, hp={my_state[12]}")
    print(f"  敌方: pos={enemy_state[0:3]}, rot={enemy_state[3:6]}, vel={enemy_state[6:9]}, hp={enemy_state[12]}")
    print(f"  is_done={is_done}")

    my_hp = my_state[12]
    enemy_hp = enemy_state[12]
    print(f"  [注意] 原始血量值: 己方={my_hp}, 敌方={enemy_hp} (可能已归一化)")
    print(f"  [注意] 原始速度值: 己方={my_state[6:9]}, 敌方={enemy_state[6:9]}")

    print("\n[步骤4] 执行50步诊断...")
    print("-" * 80)
    print(f"{'步':>4s} {'己方HP':>8s} {'敌方HP':>8s} {'距离':>8s} {'伤害':>8s} {'追近':>8s} {'步罚':>8s} {'高低':>8s} {'总奖励':>8s}")
    print("-" * 80)

    total_reward = 0.0
    damage_events = 0
    min_dist = float('inf')

    for step_idx in range(50):
        prev_my = my_state.copy()
        prev_enemy = enemy_state.copy()

        test_action = np.array([1.0, 0.0, 0.1, 0.1], dtype=np.float64)
        real_action = action.marshal_action(test_action)
        send_pack = np.append(real_action, 0.0)
        net.send_action_packet(send_pack)

        raw_obs = net.get_observation_packet()
        my_state = raw_obs[0:13].astype(np.float64)
        enemy_state = raw_obs[13:26].astype(np.float64)
        is_done = raw_obs[26]

        comps = reward.reward_components(prev_my, prev_enemy, my_state, enemy_state)
        total_reward += comps["total"]

        dist = np.linalg.norm(enemy_state[0:3] - my_state[0:3])
        min_dist = min(min_dist, dist)

        if comps["damage_reward"] != 0:
            damage_events += 1

        print(f"{step_idx+1:4d} {my_state[12]:8.1f} {enemy_state[12]:8.1f} {dist:8.1f} "
              f"{comps['damage_reward']:8.1f} {comps['approach_reward']:8.3f} "
              f"{comps['step_penalty']:8.3f} {comps['altitude_penalty']:8.1f} {comps['total']:8.3f}")

        if is_done:
            print(f"\n  [警告] 第{step_idx+1}步提前终止!")
            break

    print("-" * 80)
    print(f"\n[统计]")
    print(f"  总奖励: {total_reward:.2f}")
    print(f"  平均奖励/步: {total_reward/50:.4f}")
    print(f"  最小距离: {min_dist:.1f} (10m单位)")
    print(f"  伤害触发次数: {damage_events}/50")
    print(f"  最终己方HP: {my_state[12]:.0f}")
    print(f"  最终敌方HP: {enemy_state[12]:.0f}")

    print(f"\n[步骤5] 观测归一化范围检查...")
    net.reconnect()
    init_packet2 = np.array([114514, 1919810], dtype=np.int32)
    init_packet2 = np.append(init_packet2, initialize.generate_initial_state().astype(np.int32))
    net.send_initial_packet(init_packet2)
    raw_obs = net.get_observation_packet()
    my_s = raw_obs[0:13].astype(np.float64)
    enemy_s = raw_obs[13:26].astype(np.float64)
    proc_obs = observation.marshal_observation(my_s, enemy_s)
    print(f"  处理后观测: {proc_obs}")
    print(f"  范围: [{proc_obs.min():.4f}, {proc_obs.max():.4f}]")
    if np.all(proc_obs >= -1.5) and np.all(proc_obs <= 1.5):
        print("  [OK] 归一化范围合理")
    else:
        print("  [警告] 归一化范围超出预期，请检查!")

    net.socket.close()
    print(f"\n{'=' * 60}")
    print("诊断完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
