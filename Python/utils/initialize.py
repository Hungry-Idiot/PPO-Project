import numpy as np


def generate_initial_state():
    """
    Junior 战机 vs Fixed 靶机的初始状态。

    平台距离单位：1 unit = 10m。
    敌机放在 x=100.0，表示初始距离 1000m，满足作业要求。

    Junior 受重力影响，己方战机必须给足初速度，且高度不能为 0。

    每架飞机 12 维：
    [x, y, z, roll, pitch, yaw, u, v, w, p, q, r]
    """

    my_initial_state = np.array([
        0.0, 0.0, 1000.0,      # position: x, y, z
        0.0, 0.0, 0.0,         # attitude: roll, pitch, yaw
        30.0, 0.0, 0.0,        # velocity: u, v, w
        0.0, 0.0, 0.0          # angular velocity: p, q, r
    ], dtype=np.float64)

    enemy_initial_state = np.array([
        100.0, 0.0, 1000.0,    # 100 units = 1000m
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,         # fixed target
        0.0, 0.0, 0.0
    ], dtype=np.float64)

    return np.append(my_initial_state, enemy_initial_state)