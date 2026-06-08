import numpy as np

def generate_initial_state():
    """
    初始化双方飞机的状态。
    状态数组维度说明 (每架飞机 12 维):
    [0:3] 位置: x, y, z (高度 z 一定要大于0，否则会直接坠毁)
    [3:6] 姿态: roll(滚转), pitch(俯仰), yaw(偏航) (单位: 弧度)
    [6:9] 线速度: u(机头指向), v(侧向), w(垂直) (一定要给向前的初速度 u)
    [9:12] 角速度: p, q, r
    """
    
    # 己方战机：设定高度 z=1000.0，向前的初速度 u=200.0，其余为 0
    my_initial_state = np.array([
        0.0, 0.0, 1000.0,      # 位置
        0.0, 0.0, 0.0,         # 姿态
        5.0, 0.0, 0.0,         # 线速度 (50m/s，足够慢可盘旋缠斗)
        0.0, 0.0, 0.0          # 角速度
    ], dtype=np.float64)

    # 敌方移动靶（Simple）：初速度制造横向移动，考验 agent 追踪能力
    enemy_initial_state = np.array([
        20.0, 0.0, 1000.0,     # 位置 (200m)
        0.0, 0.0, 0.0,         # 姿态
        5.0, 3.0, 0.0,         # 线速度 (前50m/s + 侧30m/s)
        0.0, 0.0, 0.0          # 角速度
    ], dtype=np.float64)
    
    # 严格按照原版模板的格式，将两个 12 维数组拼接成一个 24 维的数组返回
    initial_state = np.append(my_initial_state, enemy_initial_state)
    
    return initial_state