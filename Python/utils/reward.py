import numpy as np

def reward_components(prev_my_state, prev_enemy_state, my_state, enemy_state):
    comps = {}

    # 1. damage reward - 最核心，保持高权重
    damage_dealt = (prev_enemy_state[12] - enemy_state[12]) * 1000.0
    damage_taken = (prev_my_state[12] - my_state[12]) * 1000.0
    comps["damage_reward"] = (damage_dealt * 2.0) - (damage_taken * 1.5)

    # 2. heading reward - 主导引导信号，去掉阈值，纯点积
    rel_pos = np.array(enemy_state[0:3]) - np.array(my_state[0:3])
    rel_dist = np.linalg.norm(rel_pos)
    if rel_dist > 1e-6:
        rel_dir = rel_pos / rel_dist
        roll, pitch, yaw = my_state[3], my_state[4], my_state[5]
        forward_x = np.cos(yaw) * np.cos(pitch)
        forward_y = np.sin(yaw) * np.cos(pitch)
        forward_z = np.sin(pitch)
        forward = np.array([forward_x, forward_y, forward_z])
        heading_dot = np.dot(forward, rel_dir)
        comps["heading_reward"] = heading_dot * 2.0
    else:
        comps["heading_reward"] = 2.0

    # 3. approach reward - 靠近奖励，权重加大
    prev_dist = np.linalg.norm(np.array(prev_enemy_state[0:3]) - np.array(prev_my_state[0:3]))
    curr_dist = np.linalg.norm(np.array(enemy_state[0:3]) - np.array(my_state[0:3]))
    comps["approach_reward"] = (prev_dist - curr_dist) * 2.0

    # 4. speed reward - 降到辅助角色，不抢主导
    speed = np.linalg.norm(my_state[6:9])
    if speed < 50.0:
        comps["speed_reward"] = -1.0
    elif speed < 100.0:
        comps["speed_reward"] = (speed - 50.0) / 50.0 * 0.2 - 0.2
    else:
        comps["speed_reward"] = 0.2

    # 5. step penalty
    comps["step_penalty"] = -0.02

    # 6. altitude penalty
    if my_state[2] < 50.0:
        comps["altitude_penalty"] = -10.0
    else:
        comps["altitude_penalty"] = 0.0

    comps["total"] = sum(comps.values())
    return comps

def calculate_reward(prev_my_state, prev_enemy_state, my_state, enemy_state):
    return reward_components(prev_my_state, prev_enemy_state, my_state, enemy_state)["total"]
