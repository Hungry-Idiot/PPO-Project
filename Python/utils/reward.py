import numpy as np

TARGET_DIST = 20.0  # 目标距离 200m，武器舒适射程


def reward_components(prev_my_state, prev_enemy_state, my_state, enemy_state):
    comps = {}

    # 1. damage reward - 最核心
    damage_dealt = (prev_enemy_state[12] - enemy_state[12]) * 1000.0
    damage_taken = (prev_my_state[12] - my_state[12]) * 1000.0
    comps["damage_reward"] = (damage_dealt * 2.0) - (damage_taken * 1.5)

    # 2. distance hold reward - 核心改动：引导保持在目标距离
    curr_dist = np.linalg.norm(
        np.array(enemy_state[0:3]) - np.array(my_state[0:3])
    )
    dist_error = abs(curr_dist - TARGET_DIST)
    comps["distance_hold_reward"] = 2.0 / (1.0 + dist_error * 0.2)

    # 3. heading reward - 距离挂钩
    rel_pos = np.array(enemy_state[0:3]) - np.array(my_state[0:3])
    if curr_dist > 1e-6:
        rel_dir = rel_pos / curr_dist
        roll, pitch, yaw = my_state[3], my_state[4], my_state[5]
        forward_x = np.cos(yaw) * np.cos(pitch)
        forward_y = np.sin(yaw) * np.cos(pitch)
        forward_z = np.sin(pitch)
        forward = np.array([forward_x, forward_y, forward_z])
        heading_dot = np.dot(forward, rel_dir)
        if curr_dist > 30.0:
            comps["heading_reward"] = heading_dot * 2.0
        else:
            comps["heading_reward"] = heading_dot * 0.5
    else:
        comps["heading_reward"] = 0.5

    # 4. proximity reward - 扩大阈值 (30=300m, 50=500m)
    if curr_dist < 30.0:
        comps["proximity_reward"] = 0.5
    elif curr_dist < 50.0:
        comps["proximity_reward"] = 0.2
    else:
        comps["proximity_reward"] = 0.0

    # 5. speed reward - 奖励低速 (5-20 = 50-200m/s)
    speed = np.linalg.norm(my_state[6:9])
    if speed < 5.0:
        comps["speed_reward"] = -0.5
    elif speed < 20.0:
        comps["speed_reward"] = 0.3
    elif speed < 40.0:
        comps["speed_reward"] = (speed - 20.0) / 20.0 * -0.2 + 0.3
    else:
        comps["speed_reward"] = -0.5

    # 6. step penalty
    comps["step_penalty"] = -0.02

    # 7. altitude penalty
    if my_state[2] < 50.0:
        comps["altitude_penalty"] = -10.0
    else:
        comps["altitude_penalty"] = 0.0

    comps["total"] = sum(comps.values())
    return comps


def calculate_reward(prev_my_state, prev_enemy_state, my_state, enemy_state):
    return reward_components(prev_my_state, prev_enemy_state, my_state, enemy_state)["total"]
