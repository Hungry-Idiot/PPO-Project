import numpy as np


def marshal_observation(my_state, enemy_state):
    """
    将平台返回的双方 13 维状态处理成 PPO 使用的 21 维 observation。

    my_state / enemy_state:
    [x, y, z, roll, pitch, yaw, u, v, w, p, q, r, hp]

    输出 21 维：
    rel_pos(3)
    dist(1)
    my_rot(3)
    my_vel(3)
    my_ang_vel(3)
    my_hp(1)
    enemy_hp(1)
    forward_vec(3)
    rel_dir(3)
    """

    my_pos = np.array(my_state[0:3], dtype=np.float64)
    enemy_pos = np.array(enemy_state[0:3], dtype=np.float64)

    # 1. 相对位置与距离
    rel_pos = enemy_pos - my_pos
    distance = np.linalg.norm(rel_pos)

    # simple fixed 1000m 阶段：
    # 初始距离 100 units = 1000m
    # 用 500 做归一化尺度，使初始距离约为 0.2
    norm_rel_pos = rel_pos / 500.0
    norm_distance = distance / 500.0

    # 2. 自身姿态、速度、角速度
    my_rot = np.array(my_state[3:6], dtype=np.float64) / np.pi
    my_vel = np.array(my_state[6:9], dtype=np.float64) / 100.0
    my_ang_vel = np.array(my_state[9:12], dtype=np.float64) / 10.0

    # 3. 血量，平台一般返回 [0, 1]
    my_hp = float(my_state[12])
    enemy_hp = float(enemy_state[12])

    # 4. 自身机头方向单位向量
    roll, pitch, yaw = my_state[3], my_state[4], my_state[5]
    forward_x = np.cos(yaw) * np.cos(pitch)
    forward_y = np.sin(yaw) * np.cos(pitch)
    forward_z = np.sin(pitch)
    forward_vec = np.array([forward_x, forward_y, forward_z], dtype=np.float64)

    # 5. 敌机相对方向单位向量
    if distance > 1e-6:
        rel_dir = rel_pos / distance
    else:
        rel_dir = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    agent_state = np.concatenate([
        norm_rel_pos,          # 3
        [norm_distance],       # 1
        my_rot,                # 3
        my_vel,                # 3
        my_ang_vel,            # 3
        [my_hp],               # 1
        [enemy_hp],            # 1
        forward_vec,           # 3
        rel_dir,               # 3
    ]).astype(np.float64)

    # 防御性检查：TrainEnv 中 observation_space 是 21 维
    assert agent_state.shape == (21,), f"observation shape error: {agent_state.shape}"

    # 防止极端异常值破坏 PPO
    agent_state = np.nan_to_num(agent_state, nan=0.0, posinf=1.0, neginf=-1.0)

    return agent_state