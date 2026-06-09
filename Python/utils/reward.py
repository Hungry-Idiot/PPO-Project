import numpy as np


# 距离单位为 10m
# 30 表示约 300m
TARGET_DIST = 40.0

# Junior 初始高度
TARGET_ALTITUDE = 1000.0

# Junior 第一版速度目标区间
# 诊断中 throttle=0.4~0.6 时速度大约进入十几到二十左右的量级，
# 因此先不要沿用 Simple 的低速偏好。
IDEAL_SPEED_MIN = 12.0
IDEAL_SPEED_MAX = 22.0


def _forward_vector_from_state(state):
    """
    根据 roll, pitch, yaw 计算机头方向。
    当前只使用 pitch/yaw，和 observation.py / Simple reward 中保持一致。
    """
    roll, pitch, yaw = state[3], state[4], state[5]

    forward_x = np.cos(yaw) * np.cos(pitch)
    forward_y = np.sin(yaw) * np.cos(pitch)
    forward_z = np.sin(pitch)

    return np.array([forward_x, forward_y, forward_z], dtype=np.float64)


def reward_components(prev_my_state, prev_enemy_state, my_state, enemy_state):
    comps = {}

    my_pos = np.array(my_state[0:3], dtype=np.float64)
    enemy_pos = np.array(enemy_state[0:3], dtype=np.float64)
    rel_pos = enemy_pos - my_pos
    curr_dist = float(np.linalg.norm(rel_pos))

    # 1. damage reward
    # 伤害仍然是最主要正反馈。
    damage_dealt = max(0.0, (prev_enemy_state[12] - enemy_state[12]) * 1000.0)
    damage_taken = max(0.0, (prev_my_state[12] - my_state[12]) * 1000.0)
    comps["damage_reward"] = damage_dealt * 4.0 - damage_taken * 3.0

    # 2. distance hold reward
    # 鼓励接近武器有效距离，但不要求贴脸。
    dist_error = abs(curr_dist - TARGET_DIST)
    comps["distance_hold_reward"] = 4.0 / (1.0 + dist_error * 0.12)

    # 3. heading reward
    # 鼓励机头指向敌机。
    if curr_dist > 1e-6:
        rel_dir = rel_pos / curr_dist
        forward = _forward_vector_from_state(my_state)
        heading_dot = float(np.dot(forward, rel_dir))
        comps["heading_reward"] = heading_dot * 4.0
    else:
        comps["heading_reward"] = 0.0

    # 4. proximity penalty
    # Junior 阶段仍然避免贴脸碰撞。
    if curr_dist < 8.0:
        comps["proximity_reward"] = -40.0
    elif curr_dist < 15.0:
        comps["proximity_reward"] = -15.0
    elif curr_dist < 25.0:
        comps["proximity_reward"] = -4.0
    else:
        comps["proximity_reward"] = 0.0

    prev_dist = float(
        np.linalg.norm(
            np.array(prev_enemy_state[0:3], dtype=np.float64)
            - np.array(prev_my_state[0:3], dtype=np.float64)
        )
    )

    if curr_dist < TARGET_DIST and curr_dist < prev_dist:
        comps["closing_too_close_penalty"] = -2.0
    else:
        comps["closing_too_close_penalty"] = 0.0

    # 5. desertion penalty
    # 防止飞远。比 Simple 稍微放宽一点，但仍然强约束。
    if curr_dist > 200.0:
        comps["desertion_penalty"] = -120.0
    elif curr_dist > 100.0:
        comps["desertion_penalty"] = -8.0
    elif curr_dist > 60.0:
        comps["desertion_penalty"] = -2.0
    else:
        comps["desertion_penalty"] = 0.0

    # 6. speed reward
    # Junior 不再奖励过低速度；速度太低可能无法维持飞行。
    speed = float(np.linalg.norm(my_state[6:9]))

    if IDEAL_SPEED_MIN <= speed <= IDEAL_SPEED_MAX:
        comps["speed_reward"] = 1.0
    elif 8.0 <= speed < IDEAL_SPEED_MIN:
        comps["speed_reward"] = -0.5
    elif speed < 8.0:
        comps["speed_reward"] = -3.0
    elif IDEAL_SPEED_MAX < speed <= 32.0:
        comps["speed_reward"] = 0.2
    else:
        comps["speed_reward"] = -3.0

    # 7. altitude hold reward / penalty
    # Junior 受重力影响，高度保持必须明确加入奖励。
    altitude = float(my_state[2])
    altitude_error = abs(altitude - TARGET_ALTITUDE)

    comps["altitude_hold_reward"] = 1.0 / (1.0 + altitude_error * 0.02)

    if altitude < 600.0:
        comps["altitude_penalty"] = -150.0
    elif altitude < 800.0:
        comps["altitude_penalty"] = -40.0
    elif altitude < 900.0:
        comps["altitude_penalty"] = -8.0
    elif altitude > 1300.0:
        comps["altitude_penalty"] = -5.0
    else:
        comps["altitude_penalty"] = 0.0

    # 8. vertical speed penalty
    # my_state[8] 是 z 方向速度。负值过大说明正在快速掉高。
    vertical_speed = float(my_state[8])

    if vertical_speed < -10.0:
        comps["vertical_speed_penalty"] = -8.0
    elif vertical_speed < -5.0:
        comps["vertical_speed_penalty"] = -3.0
    else:
        comps["vertical_speed_penalty"] = 0.0

    # 9. step penalty
    comps["step_penalty"] = -0.02

    # 10. death penalty
    # 继续使用 transition-only，避免死亡后重复扣分。
    death_transition = prev_my_state[12] > 0.01 and my_state[12] <= 0.01

    if death_transition:
        comps["death_penalty"] = -2000.0
    else:
        comps["death_penalty"] = 0.0

    # 11. kill bonus
    kill_transition = prev_enemy_state[12] > 0.01 and enemy_state[12] <= 0.01

    if kill_transition:
        comps["kill_bonus"] = 3000.0
    else:
        comps["kill_bonus"] = 0.0

    comps["total"] = sum(comps.values())

    return comps


def calculate_reward(prev_my_state, prev_enemy_state, my_state, enemy_state):
    return reward_components(
        prev_my_state,
        prev_enemy_state,
        my_state,
        enemy_state
    )["total"]