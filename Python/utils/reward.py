import numpy as np


TARGET_DIST = 20.0        # 200m，期望攻击距离
FIRE_DIST_MIN = 3.0       # 30m，太近容易穿模/贴脸
FIRE_DIST_MAX = 100.0     # 1000m，Junior 阶段先鼓励较早进入攻击窗口

MAX_DIST = 500.0          # 5000m，飞远阈值
LOW_ALT_WARN = 700.0
LOW_ALT_DANGER = 400.0
LOW_ALT_CRASH = 200.0


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

    # 1. 伤害奖励：Junior 阶段依然以伤害为主目标
    damage_dealt = max(0.0, (prev_enemy_state[12] - enemy_state[12]) * 1000.0)
    damage_taken = max(0.0, (prev_my_state[12] - my_state[12]) * 1000.0)
    comps["damage_reward"] = damage_dealt * 12.0 - damage_taken * 1.0

    # 2. 势能式接近奖励：只奖励距离变小
    approach_delta = prev_dist - dist
    comps["approach_reward"] = np.clip(approach_delta, -5.0, 5.0) * 8.0

    # 3. 距离保持奖励：靠近攻击距离更好，但权重不要过大
    dist_error = abs(dist - TARGET_DIST)
    comps["distance_hold_reward"] = 2.0 / (1.0 + 0.08 * dist_error)

    # 4. 机头朝向奖励
    if dist > 1e-6:
        rel_dir = rel_pos / dist
        forward = _forward_vector(my_state)
        heading_dot = float(np.dot(forward, rel_dir))
    else:
        heading_dot = 0.0

    comps["heading_reward"] = heading_dot * 5.0

    # 5. 进入攻击窗口且对准时奖励
    if FIRE_DIST_MIN <= dist <= FIRE_DIST_MAX and heading_dot > 0.95:
        comps["aim_in_range_reward"] = 20.0
    elif FIRE_DIST_MIN <= dist <= FIRE_DIST_MAX and heading_dot > 0.80:
        comps["aim_in_range_reward"] = 8.0
    else:
        comps["aim_in_range_reward"] = 0.0

    enemy_hp = float(enemy_state[12])
    prev_enemy_hp = float(prev_enemy_state[12])
    enemy_damaged = enemy_hp < 0.99

    if enemy_damaged and dist > 240.0:
        comps["post_damage_disengage_penalty"] = -150.0
    elif enemy_damaged and dist > 180.0:
        comps["post_damage_disengage_penalty"] = -80.0
    elif enemy_damaged and dist > 120.0:
        comps["post_damage_disengage_penalty"] = -30.0
    else:
        comps["post_damage_disengage_penalty"] = 0.0

    # 6. 太近惩罚
    if dist < 3.0:
        comps["proximity_penalty"] = -30.0
    elif dist < 6.0:
        comps["proximity_penalty"] = -8.0
    else:
        comps["proximity_penalty"] = 0.0

    # 7. 飞远惩罚
    if dist > MAX_DIST:
        comps["desertion_penalty"] = -120.0
    elif dist > 350.0:
        comps["desertion_penalty"] = -30.0
    elif dist > 200.0:
        comps["desertion_penalty"] = -5.0
    else:
        comps["desertion_penalty"] = 0.0

    # 8. 速度保护：Junior 需要能量，但也不能过快
    speed = float(np.linalg.norm(my_state[6:9]))
    if speed < 10.0:
        comps["speed_reward"] = -30.0
    elif speed < 20.0:
        comps["speed_reward"] = -5.0
    elif speed <= 80.0:
        comps["speed_reward"] = 0.5
    elif speed <= 120.0:
        comps["speed_reward"] = -3.0
    else:
        comps["speed_reward"] = -15.0

    # 9. 高度保护：Junior 受重力影响，这一项比 Simple 更重要
    altitude = float(my_state[2])
    if altitude < LOW_ALT_CRASH:
        comps["altitude_penalty"] = -200.0
    elif altitude < LOW_ALT_DANGER:
        comps["altitude_penalty"] = -60.0
    elif altitude < LOW_ALT_WARN:
        comps["altitude_penalty"] = -10.0
    else:
        comps["altitude_penalty"] = 0.0

    # 10. 垂直速度保护，避免持续高速下坠
    vertical_speed = float(my_state[8])
    if vertical_speed < -40.0:
        comps["vertical_speed_penalty"] = -40.0
    elif vertical_speed < -20.0:
        comps["vertical_speed_penalty"] = -10.0
    else:
        comps["vertical_speed_penalty"] = 0.0

    # 11. 姿态稳定惩罚，避免大 pitch / 大 roll 长时间失控
    roll = float(my_state[3])
    pitch = float(my_state[4])

    roll_excess = max(0.0, abs(roll) - 1.2)
    pitch_excess = max(0.0, abs(pitch) - 0.7)
    comps["attitude_penalty"] = -5.0 * roll_excess - 8.0 * pitch_excess

    # 12. 每步惩罚，鼓励尽快击杀
    comps["step_penalty"] = -0.05

    # 13. 死亡惩罚
    death_transition = prev_my_state[12] > 0.01 and my_state[12] <= 0.01
    comps["death_penalty"] = -3000.0 if death_transition else 0.0

    # 14. 击杀奖励
    kill_transition = prev_enemy_hp > 0.01 and enemy_hp <= 0.01
    comps["kill_bonus"] = 15000.0 if kill_transition else 0.0

    comps["total"] = sum(comps.values())
    return comps


def calculate_reward(prev_my_state, prev_enemy_state, my_state, enemy_state):
    return reward_components(prev_my_state, prev_enemy_state, my_state, enemy_state)["total"]