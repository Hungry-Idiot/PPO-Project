import numpy as np

TARGET_DIST = 20.0  # 目标距离 200m，武器舒适射程

def reward_components(prev_my_state, prev_enemy_state, my_state, enemy_state):
    comps = {}

    # 1. damage reward - 伤害是第一生产力
    damage_dealt = (prev_enemy_state[12] - enemy_state[12]) * 1000.0
    damage_taken = (prev_my_state[12] - my_state[12]) * 1000.0
    comps["damage_reward"] = (damage_dealt * 4.0) - (damage_taken * 1.0)

    # 2. distance hold reward - 引导保持在目标距离
    curr_dist = np.linalg.norm(np.array(enemy_state[0:3]) - np.array(my_state[0:3]))
    dist_error = abs(curr_dist - TARGET_DIST)
    comps["distance_hold_reward"] = 2.0 / (1.0 + dist_error * 0.2)

    # 3. heading reward - 核心修复：永远鼓励机头对敌！取消靠近后的惩罚
    rel_pos = np.array(enemy_state[0:3]) - np.array(my_state[0:3])
    if curr_dist > 1e-6:
        rel_dir = rel_pos / curr_dist
        roll, pitch, yaw = my_state[3], my_state[4], my_state[5]
        forward_x = np.cos(yaw) * np.cos(pitch)
        forward_y = np.sin(yaw) * np.cos(pitch)
        forward_z = np.sin(pitch)
        forward = np.array([forward_x, forward_y, forward_z])
        heading_dot = np.dot(forward, rel_dir)
        
        # 只要对准就给大额奖励，鼓励死咬不放
        comps["heading_reward"] = heading_dot * 2.0
    else:
        comps["heading_reward"] = 0.0

    # 4. proximity penalty - 缩小恐惧圈，只惩罚真正要相撞的贴脸距离
    if curr_dist < 4.0:
        comps["proximity_reward"] = -10.0
    elif curr_dist < 7.0:
        comps["proximity_reward"] = -2.0
    else:
        comps["proximity_reward"] = 0.0

    # 5. desertion penalty - 新增逃脱惩罚！严禁打完就跑
    if curr_dist > 150.0:
        comps["desertion_penalty"] = -100.0
    elif curr_dist > 50.0:
        comps["desertion_penalty"] = -2.0
    else:
        comps["desertion_penalty"] = 0.0
        
    # 6. speed reward - 靶机速度约为 5.8 (58m/s)，鼓励战机保持 50~100m/s 的追击速度
    speed = np.linalg.norm(my_state[6:9])
    if speed > 15.0:
        comps["speed_reward"] = -3.0
    elif speed > 12.0:
        comps["speed_reward"] = -1.0
    elif 3.0 <= speed <= 12.0:
        comps["speed_reward"] = 1.0
    else:
        comps["speed_reward"] = -0.5

    # 7. step penalty
    comps["step_penalty"] = -0.02

    # 8. altitude penalty - 稍微降低一点高度惩罚的敏感度，允许俯冲攻击
    if my_state[2] < 30.0:    
        comps["altitude_penalty"] = -20.0
    elif my_state[2] < 50.0:  
        comps["altitude_penalty"] = -5.0
    else:
        comps["altitude_penalty"] = 0.0

    # 9. death penalty - 继续保持一票否决自杀流
    death_transition = prev_my_state[12] > 0.01 and my_state[12] <= 0.01

    if death_transition:
        comps["death_penalty"] = -1000.0
    else:
        comps["death_penalty"] = 0.0

    kill_transition = prev_enemy_state[12] > 0.01 and enemy_state[12] <= 0.01

    if kill_transition:
        comps["kill_bonus"] = 3000.0
    else:
        comps["kill_bonus"] = 0.0

    comps["total"] = sum(comps.values())
    return comps

def calculate_reward(prev_my_state, prev_enemy_state, my_state, enemy_state):
    return reward_components(prev_my_state, prev_enemy_state, my_state, enemy_state)["total"]