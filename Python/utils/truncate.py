import numpy as np


MAX_DIST = 240.0       # 240 units = 2400m
MIN_ALTITUDE = 150.0
MAX_SPEED = 200.0


def check_truncation(my_state, enemy_state):
    my_pos = np.array(my_state[0:3], dtype=np.float64)
    enemy_pos = np.array(enemy_state[0:3], dtype=np.float64)

    dist = float(np.linalg.norm(enemy_pos - my_pos))
    altitude = float(my_state[2])
    speed = float(np.linalg.norm(my_state[6:9]))

    if dist > MAX_DIST:
        return True

    if altitude < MIN_ALTITUDE:
        return True

    if speed > MAX_SPEED:
        return True

    return False