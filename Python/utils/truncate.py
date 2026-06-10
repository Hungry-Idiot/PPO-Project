import numpy as np

MAX_DIST = 350.0  # 350 units = 3500m

def check_truncation(my_state, enemy_state):
    # 计算双方距离，单位为平台单位：1 unit = 10m
    dist = np.linalg.norm(np.array(enemy_state[0:3]) - np.array(my_state[0:3]))

    # 固定靶 1000m 阶段，初始距离就是 100 units。
    # 给足探索空间，超过 350 units 再认为飞远。
    if dist > MAX_DIST:
        return True

    return False