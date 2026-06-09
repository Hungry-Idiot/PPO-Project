"""
Evaluate trained PPO model — deterministic inference against battle_server.
Watch UE5 for visual confirmation of loitering/killing behavior.
"""
import sys
import os
# 保证能找到 utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO
from utils import adaptor, observation, action, initialize

# ==========================================
MODEL_PATH = "./output/best_simple_model.zip"
CONFIG_PATH = "./config/envs.yaml"
# ==========================================

def main():
    model = PPO.load(MODEL_PATH)

    # 1. 初始化双路连接
    net_my = adaptor.NetworkAdaptor(CONFIG_PATH)
    net_my.connect()

    net_enemy = adaptor.NetworkAdaptor(CONFIG_PATH)
    net_enemy.port += 1 # 连接到靶机的 1001 端口
    net_enemy.connect()

    # 2. 发送双路初始化包
    init_state = initialize.generate_initial_state()
    my_init = init_state[0:12].astype(np.int32)
    enemy_init = init_state[12:24].astype(np.int32)

    # 我方视角打包
    init_packet_my = np.array([114514, 1919810], dtype=np.int32)
    init_packet_my = np.append(init_packet_my, my_init)
    init_packet_my = np.append(init_packet_my, enemy_init)

    # 敌方(靶机)视角打包 (数据要对调)
    init_packet_enemy = np.array([114514, 1919811], dtype=np.int32)
    init_packet_enemy = np.append(init_packet_enemy, enemy_init)
    init_packet_enemy = np.append(init_packet_enemy, my_init)

    net_my.send_initial_packet(init_packet_my)
    net_enemy.send_initial_packet(init_packet_enemy)

    # 3. 双路接收初始观测值
    raw_my = net_my.get_observation_packet()
    _ = net_enemy.get_observation_packet()

    my_state = raw_my[0:13].astype(np.float64)
    enemy_state = raw_my[13:26].astype(np.float64)
    obs = observation.marshal_observation(my_state, enemy_state)

    total_damage = 0.0
    max_damage_per_step = 0.0
    min_dist = float('inf')

    header = f"{'Step':>5s} {'Dist(m)':>8s} {'EnemyHP':>10s} {'Damage':>8s} {'TotalDmg':>9s} {'Thr':>6s} {'Pitch':>6s} {'Roll':>6s} {'Yaw':>6s} {'Speed':>7s}"
    print(header)
    print("-" * len(header))

    for step in range(1000):
        # 智能体决策
        agent_action, _ = model.predict(obs, deterministic=True)
        real_action = action.marshal_action(agent_action)
        send_pack = np.append(real_action, 0.0)
        
        # 靶机决策 (全零无操作滑行)
        enemy_action = np.zeros(4, dtype=np.float64)
        enemy_send_pack = np.append(enemy_action, 0.0)

        # 双路发送控制指令
        net_my.send_action_packet(send_pack)
        net_enemy.send_action_packet(enemy_send_pack)

        # 双路接收战斗数据
        raw_my = net_my.get_observation_packet()
        _ = net_enemy.get_observation_packet()

        prev_enemy_hp = enemy_state[12]
        my_state = raw_my[0:13].astype(np.float64)
        enemy_state = raw_my[13:26].astype(np.float64)
        is_done = raw_my[26]

        obs = observation.marshal_observation(my_state, enemy_state)

        damage = max(0.0, (prev_enemy_hp - enemy_state[12]) * 1000.0)
        total_damage += damage
        max_damage_per_step = max(max_damage_per_step, damage)
        dist = np.linalg.norm(enemy_state[0:3] - my_state[0:3]) * 10.0
        min_dist = min(min_dist, dist)

        speed = np.linalg.norm(my_state[6:9])

        print(f"{step+1:5d} {dist:8.1f} {enemy_state[12]:10.4f} {damage:8.1f} {total_damage:9.1f} "
              f"{real_action[0]:6.3f} {real_action[1]:6.3f} {real_action[2]:6.3f} {real_action[3]:6.3f} "
              f"{speed:7.1f}")

        if is_done:
            print(f"\n[DONE] Episode terminated at step {step+1}")
            break

    print(f"\n{'='*50}")
    print(f"Final stats:")
    print(f"  Steps:       {step+1}")
    print(f"  Total DMG:   {total_damage:.1f} HP")
    print(f"  Max DMG/step:{max_damage_per_step:.1f} HP")
    print(f"  Min dist:    {min_dist:.1f} m")
    print(f"  Final HP:    {enemy_state[12]:.4f} (enemy)  {my_state[12]:.4f} (self)")
    killed = enemy_state[12] <= 0.01
    print(f"  KILL:        {'YES' if killed else 'NO'}")
    print(f"{'='*50}")

    net_my.socket.close()
    net_enemy.socket.close()

if __name__ == "__main__":
    main()