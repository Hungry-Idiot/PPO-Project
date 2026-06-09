import numpy as np


# Junior 第一版动作映射
# 诊断结果显示：
# - throttle=0.4 基本平飞
# - throttle=0.5/0.6 有爬升能力
# 因此先把物理油门限制到 [0.2, 0.6]：
# - PPO 输出 -1 -> throttle = 0.2
# - PPO 输出  0 -> throttle = 0.4
# - PPO 输出 +1 -> throttle = 0.6
MIN_THROTTLE = 0.2
MAX_THROTTLE = 0.6


def marshal_action(action):
    """
    处理智能体输出的动作，映射为仿真环境需要的物理控制量。

    PPO 输出:
        action[0] throttle raw, 范围 [-1, 1]
        action[1] pitch,        范围 [-1, 1]
        action[2] roll,         范围 [-1, 1]
        action[3] yaw,          范围 [-1, 1]

    仿真平台需要:
        throttle: [0, 1]
        pitch:    [-1, 1]
        roll:     [-1, 1]
        yaw:      [-1, 1]
    """
    clipped_action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

    processed_action = clipped_action.copy().astype(np.float64)

    raw_throttle = (clipped_action[0] + 1.0) / 2.0
    processed_action[0] = MIN_THROTTLE + (MAX_THROTTLE - MIN_THROTTLE) * raw_throttle

    processed_action[1] = clipped_action[1]
    processed_action[2] = clipped_action[2]
    processed_action[3] = clipped_action[3]

    return processed_action