import numpy as np

def marshal_action(action):
    """
    处理智能体输出的动作，映射为仿真环境需要的物理控制量。
    action 数组索引对应: [0]油门, [1]升降舵(俯仰), [2]副翼(滚转), [3]方向舵(偏航)
    """
    # 1. 裁剪动作，防止神经网络在探索初期输出越界值（例如爆出 -2.5 或 1.8）
    clipped_action = np.clip(action, -1.0, 1.0)
    
    # 2. 拷贝一份用于修改，并确保数据类型为环境所需的 float64
    processed_action = clipped_action.copy().astype(np.float64)
    
    # 3. 核心映射：将油门 (索引 0) 从神经网络的 [-1, 1] 线性映射到物理引擎的 [0, 1]
    # 公式: (x + 1) / 2
    raw_throttle = (clipped_action[0] + 1.0) / 2.0

    # Simple moving target 阶段不需要满油门。
    # 限制最大油门，减少高速冲出战场。
    processed_action[0] = 0.6 * raw_throttle

    # 索引 1, 2, 3 分别是俯仰、滚转、偏航，因为它们本身就需要是 [-1, 1]，所以直接透传即可
    
    return processed_action