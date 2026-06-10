import numpy as np


def generate_initial_state():
    """
    Simple 战机 vs Fixed 靶机的初始状态。

    平台距离单位：1 unit = 10m。
    因此敌机放在 x=100.0，表示初始距离 1000m，满足作业要求。

    每架飞机 12 维：
    [x, y, z, roll, pitch, yaw, u, v, w, p, q, r]
    """

    # 己方战机：高度 1000，机头朝 +x，给一个较慢前向初速度
    my_initial_state = np.array([
        0.0, 0.0, 1000.0,      # 位置 x, y, z
        0.0, 0.0, 0.0,         # 姿态 roll, pitch, yaw
        5.0, 0.0, 0.0,         # 线速度 u, v, w
        0.0, 0.0, 0.0          # 角速度 p, q, r
    ], dtype=np.float64)

    # Fixed 靶机：放在正前方 1000m，速度为 0
    enemy_initial_state = np.array([
        100.0, 0.0, 1000.0,    # 100 units = 1000m
        0.0, 0.0, 0.0,         # 姿态
        0.0, 0.0, 0.0,         # 固定靶机，速度为 0
        0.0, 0.0, 0.0          # 角速度
    ], dtype=np.float64)

    return np.append(my_initial_state, enemy_initial_state)