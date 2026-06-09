import numpy as np


def generate_initial_state():
    """
    Junior 阶段第一版初始化。

    状态数组维度说明，每架飞机 12 维:
        [0:3]   位置: x, y, z
        [3:6]   姿态: roll, pitch, yaw，单位弧度
        [6:9]   线速度: x_v, y_v, z_v
        [9:12]  角速度: x_w, y_w, z_w

    课程平台中：
        距离相关单位为 10m
        z=1000 表示约 10000m 高度
        x=30 表示约 300m 距离
    """

    # 己方 Junior 战机
    # 诊断中发现平台第一帧速度可能不完全等于这里配置的速度，
    # 但仍然保留向前初速度，用于兼容受重力影响机型的要求。
    my_initial_state = np.array([
        0.0, 0.0, 1000.0,      # 位置 x, y, z
        0.0, 0.0, 0.0,         # 姿态 roll, pitch, yaw
        10.0, 0.0, 0.0,        # 初始线速度
        0.0, 0.0, 0.0          # 初始角速度
    ], dtype=np.float64)

    # Junior 第一阶段先打固定靶/近距靶
    # 如果房间本身是 Fixed Target，敌机动力学通常由平台目标模式决定；
    # 这里仍然发送完整 enemy 初始状态。
    enemy_initial_state = np.array([
        30.0, 0.0, 1000.0,     # 敌方位置，约 300m
        0.0, 0.0, 0.0,         # 敌方姿态
        0.0, 0.0, 0.0,         # 固定靶先不给速度
        0.0, 0.0, 0.0          # 敌方角速度
    ], dtype=np.float64)

    initial_state = np.append(my_initial_state, enemy_initial_state)

    return initial_state