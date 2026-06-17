import numpy as np


def marshal_action(action):
    clipped_action = np.clip(action, -1.0, 1.0)
    processed_action = clipped_action.copy().astype(np.float64)

    # Junior 需要保持能量，避免攻击后低速脱离。
    # throttle: [-1, 1] -> [0.40, 0.90]
    raw_throttle = (clipped_action[0] + 1.0) / 2.0
    processed_action[0] = 0.40 + 0.50 * raw_throttle

    # 增加一点姿态控制权限，帮助重新转回目标。
    processed_action[1] = 0.70 * clipped_action[1]   # pitch
    processed_action[2] = 0.90 * clipped_action[2]   # roll
    processed_action[3] = 0.75 * clipped_action[3]   # yaw

    return processed_action