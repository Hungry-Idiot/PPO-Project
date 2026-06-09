# utils/truncate.py 的逻辑示例
import numpy as np

def check_truncation(my_state, enemy_state):
    # 计算双方距离
    dist = np.linalg.norm(np.array(enemy_state[0:3]) - np.array(my_state[0:3]))
    
    # 如果距离超过 1500 米，立刻截断（返回 True）
    if dist > 150.0: # (这里假设你的距离单位是 10m，所以 150 = 1500m)
        return True
        
    return False