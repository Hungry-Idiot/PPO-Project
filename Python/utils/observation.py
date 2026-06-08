import numpy as np

# This is the observation processing function. Remember to modify the declarations in trainenv.py correspondingly.
def marshal_observation(my_state, enemy_state):
    # 转换为 numpy 数组以方便向量化计算
    my_pos = np.array(my_state[0:3])
    enemy_pos = np.array(enemy_state[0:3])
    
    # 1. 相对位置与距离特征
    rel_pos = enemy_pos - my_pos
    distance = np.linalg.norm(rel_pos)
    
    # 仿真平台中距离相关单位为10m [cite: 142]
    # 初始敌我距离约3000单位=30000m，战场范围约5000单位
    # 除以 5000.0 归一化到 [0, ~1] 范围
    norm_rel_pos = rel_pos / 5000.0
    norm_distance = distance / 5000.0
    
    # 2. 自身姿态与速度特征
    # 角度单位已经是弧度 [cite: 142]，除以 Pi 归一化到 [-1, 1] 左右
    my_rot = np.array(my_state[3:6]) / np.pi      
    my_vel = np.array(my_state[6:9]) / 100.0      # 线速度归一化
    my_ang_vel = np.array(my_state[9:12]) / 10.0  # 角速度归一化
    
    # 服务器返回血量已归一化到 [0, 1]，直接透传
    my_hp = my_state[12]
    enemy_hp = enemy_state[12]

    # 3. 自身机头方向单位向量
    roll, pitch, yaw = my_state[3], my_state[4], my_state[5]
    forward_x = np.cos(yaw) * np.cos(pitch)
    forward_y = np.sin(yaw) * np.cos(pitch)
    forward_z = np.sin(pitch)
    forward_vec = np.array([forward_x, forward_y, forward_z])

    # 4. 相对方位角 — 敌机在我前方的方向
    if distance > 1e-6:
        rel_dir = rel_pos / distance
    else:
        rel_dir = np.array([0.0, 0.0, 0.0])

    # 维度: rel_pos(3)+dist(1)+my_rot(3)+my_vel(3)+my_ang_vel(3)+my_hp(1)+enemy_hp(1)+forward(3)+rel_dir(3)=19
    agent_state = np.concatenate([
        norm_rel_pos,
        [norm_distance],
        my_rot,
        my_vel,
        my_ang_vel,
        [my_hp],
        [enemy_hp],
        forward_vec,
        rel_dir,
    ]).astype(np.float64)
    
    return agent_state