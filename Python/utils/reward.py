import numpy as np


TARGET_DIST = 20.0       # 200m，期望攻击距离
FIRE_DIST_MIN = 3.0      # 30m，太近容易贴脸/穿模
FIRE_DIST_MAX = 80.0     # 800m，进入较近攻击区
MAX_DIST = 350.0         # 3500m，飞远阈值


def _forward_vector(state):
    roll, pitch, yaw = state[3], state[4], state[5]
    return np.array([
        np.cos(yaw) * np.cos(pitch),
        np.sin(yaw) * np.cos(pitch),
        np.sin(pitch),
    ], dtype=np.float64)


def reward_components(prev_my_state, prev_enemy_state, my_state, enemy_state):
    comps = {}

    my_pos = np.array(my_state[0:3], dtype=np.float64)
    enemy_pos = np.array(enemy_state[0:3], dtype=np.float64)
    prev_my_pos = np.array(prev_my_state[0:3], dtype=np.float64)
    prev_enemy_pos = np.array(prev_enemy_state[0:3], dtype=np.float64)

    rel_pos = enemy_pos - my_pos
    prev_rel_pos = prev_enemy_pos - prev_my_pos

    dist = float(np.linalg.norm(rel_pos))
    prev_dist = float(np.linalg.norm(prev_rel_pos))

    # 1. 伤害奖励：主要正反馈
    damage_dealt = max(0.0, (prev_enemy_state[12] - enemy_state[12]) * 1000.0)
    damage_taken = max(0.0, (prev_my_state[12] - my_state[12]) * 1000.0)
    comps["damage_reward"] = damage_dealt * 5.0 - damage_taken * 1.0

    # 2. 势能式接近奖励：只奖励“距离变小”，避免停在某处刷奖励
    approach_delta = prev_dist - dist
    comps["approach_reward"] = np.clip(approach_delta, -5.0, 5.0) * 5.0

    # 3. 距离保持奖励：靠近 200m 攻击距离更好
    dist_error = abs(dist - TARGET_DIST)
    comps["distance_hold_reward"] = 2.0 / (1.0 + 0.1 * dist_error)

    # 4. 机头朝向奖励
    if dist > 1e-6:
        rel_dir = rel_pos / dist
        forward = _forward_vector(my_state)
        heading_dot = float(np.dot(forward, rel_dir))
    else:
        heading_dot = 0.0

    comps["heading_reward"] = heading_dot * 3.0

    # 5. 进入攻击区且对准时给额外奖励
    if FIRE_DIST_MIN <= dist <= FIRE_DIST_MAX and heading_dot > 0.95:
        comps["aim_in_range_reward"] = 5.0
    elif FIRE_DIST_MIN <= dist <= FIRE_DIST_MAX and heading_dot > 0.80:
        comps["aim_in_range_reward"] = 2.0
    else:
        comps["aim_in_range_reward"] = 0.0

    # 6. 太近惩罚，避免贴脸/碰撞/穿模
    if dist < 3.0:
        comps["proximity_penalty"] = -20.0
    elif dist < 6.0:
        comps["proximity_penalty"] = -5.0
    else:
        comps["proximity_penalty"] = 0.0

    # 7. 飞远惩罚
    if dist > MAX_DIST:
        comps["desertion_penalty"] = -100.0
    elif dist > 250.0:
        comps["desertion_penalty"] = -5.0
    else:
        comps["desertion_penalty"] = 0.0

    # 8. 速度奖励：鼓励不要太慢，也不要过快
    speed = float(np.linalg.norm(my_state[6:9]))
    if speed > 20.0:
        comps["speed_reward"] = -3.0
    elif speed > 15.0:
        comps["speed_reward"] = -1.0
    elif 3.0 <= speed <= 15.0:
        comps["speed_reward"] = 1.0
    else:
        comps["speed_reward"] = -0.5

    # 9. 高度保护。Simple 不受重力，但仍避免贴地
    if my_state[2] < 30.0:
        comps["altitude_penalty"] = -20.0
    elif my_state[2] < 50.0:
        comps["altitude_penalty"] = -5.0
    else:
        comps["altitude_penalty"] = 0.0

    # 10. 每步惩罚，鼓励尽快击杀
    comps["step_penalty"] = -0.03

    # 11. 死亡惩罚，只在死亡转移瞬间扣一次
    death_transition = prev_my_state[12] > 0.01 and my_state[12] <= 0.01
    comps["death_penalty"] = -1000.0 if death_transition else 0.0

    # 12. 击杀奖励，只在击杀转移瞬间给一次
    kill_transition = prev_enemy_state[12] > 0.01 and enemy_state[12] <= 0.01
    comps["kill_bonus"] = 4000.0 if kill_transition else 0.0

    comps["total"] = sum(comps.values())
    return comps


def calculate_reward(prev_my_state, prev_enemy_state, my_state, enemy_state):
    return reward_components(prev_my_state, prev_enemy_state, my_state, enemy_state)["total"]