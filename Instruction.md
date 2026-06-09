# 课程大作业 —— PPO 强化学习空战智能体训练

## 1. 项目目标

使用 **PPO（Proximal Policy Optimization, Stable-Baselines3）** 训练战斗机智能体，在 UE5 仿真中击败敌方战机（**Fixed Target → Simple 移动靶，渐进升级**）。

## 2. 系统架构

```
Python Agent (PPO/SB3)
  └─ TCP Socket (127.0.0.1:1000)
       └─ Docker battle_server (端口 8887, 战斗端口 1000-2000)
            └─ UE5 StudentClient (可视化渲染 + 物理仿真)
```

- **Docker battle_server**：战斗逻辑，管理房间和战斗生命周期
- **UE5 StudentClient**：3D 物理仿真 + 渲染，~29 FPS
- **Python Agent**：通过 TCP Socket 每步发送控制、接收观测

**启动顺序**：Docker battle_server → UE5 创建房间（端口 1000）→ Python 训练脚本

## 3. 通信协议

### 3.1 初始包 InitData（400 bytes）
```python
INITIAL_PACKET_FORMAT = "<26i296x"
# [room(1), unit(1), my_state(12), enemy_state(12)] → 26 int32
# room=114514, unit=1919810
```
双方初始 12 维状态：位置3 + 姿态3 + 线速度3 + 角速度3

### 3.2 观测包 BattleData（216 bytes, 27 doubles）
```python
GETTING_PACKET_FORMAT = "=27d"
# [my_state(13), enemy_state(13), is_done(1)]
```
每架飞机 13 维：[位置x3, 姿态x3, 线速度x3, 角速度x3, 血量HP]
- 血量已归一化到 [0, 1]
- is_done: 0.0 = 继续, 1.0 = 终止
- 距离单位: **10m**（值 100 = 1000m），角度单位: **弧度**

### 3.3 控制包 CtrlData（40 bytes, 5 doubles）
```python
SENDING_PACKET_FORMAT = "<5d"
# [throttle, pitch, roll, yaw, truncation]
```
- throttle: [0, 1]，实际油门（训练侧映射 [-1,1]→[0,1]）
- pitch/roll/yaw: [-1, 1]，操纵面偏转
- truncation: 0.0/1.0，截断标志
- **无显式开火指令** —— 武器自动：攻击范围与受击体积重叠 → 固定伤害

## 4. Python 代码结构

```
Python/
├── main.py              # 训练入口，PPO 超参
├── evaluate.py          # 模型评估：加载训练权重，确定性推理，观察 UE5 行为
├── config/envs.yaml     # host: 127.0.0.1, port: 1000
├── envs/train_env.py    # Gymnasium Env (TrainEnv)
├── utils/
│   ├── adaptor.py       # TCP 网络适配器
│   ├── action.py        # 动作映射 [throttle, pitch, roll, yaw]
│   ├── observation.py   # 观测处理 21维
│   ├── reward.py        # 奖励函数（核心，已重写多轮）
│   ├── initialize.py    # 初始状态生成
│   ├── truncate.py      # 截断检查（恒为 False）
│   └── callback.py      # 训练回调 + 奖励分量 CSV 日志
├── diagnose_env.py      # 环境连通性诊断
├── diagnose_combat.py   # 交战距离诊断（PID 导引）
├── diagnose_damage_short.py  # 武器伤害验证（超近距离 + 零操作）
├── diagnose_weapon_range.py  # 武器攻击范围测量（单连接二分搜索）
└── diagnose_simple_vs_simple.py  # Simple vs Simple 双连接诊断
```

### 4.1 观测空间（21 维）
```python
agent_state = concat([
    norm_rel_pos(3),     # 相对位置 / 5000
    norm_dist(1),        # 距离 / 5000
    my_rot(3),           # 姿态角 / π
    my_vel(3),           # 线速度 / 100
    my_ang_vel(3),       # 角速度 / 10
    my_hp(1),            # [0,1]
    enemy_hp(1),         # [0,1]
    forward_vec(3),      # 机头方向单位向量
    rel_dir(3),          # 相对方位单位向量
])
```

### 4.2 动作空间（4 维）
```python
action = [throttle, pitch, roll, yaw]  # 全部 ∈ [-1, 1]
# throttle: [-1,1] → [0,1]  公式 (x+1)/2
# pitch/roll/yaw: 直接透传
```

### 4.3 奖励函数（reward.py — 当前版本，移动靶优化）
```python
TARGET_DIST = 20.0  # 目标保持距离 200m

# 9 个分量，面向移动靶追瞄作战：
damage_reward:         (damage_dealt * 2.0) - (damage_taken * 1.5)
distance_hold_reward:  2.0 / (1.0 + dist_error * 0.2)   # 保持 ~200m 最佳射程
heading_reward:        heading_dot * 2.0                  # 永远鼓励机头对敌（取消近距离弱化）
proximity_reward:      <4(=40m)=-20, <7(=70m)=-5, else 0 # 仅惩罚贴脸防撞
desertion_penalty:     >150(1500m)=-1000, >50(500m)=-10  # 逃脱判负，严禁飞离战场
speed_reward:          5-9(50-90m/s)=1.0, >9=-2, >12=-10 # 追瞄转弯速度 50-90m/s
step_penalty:          -0.02
altitude_penalty:      <30=-20, <50=-5, else 0           # 允许有限俯冲攻击
death_penalty:         -1000 if HP ≤ 0                    # 一票否决自杀流
```

### 4.4 初始状态（initialize.py — 当前版本，Simple 移动靶）
```python
my:    pos=(0,0,1000), forward_vel=5    # 50m/s，足够慢可盘旋缠斗
enemy: pos=(20,0,1000), vel=(5,3,0)     # 初始相距 200m，前50m/s + 侧30m/s 移动靶
```

### 4.5 PPO 超参（main.py — 当前版本）
```python
n_steps=512, batch_size=128, n_epochs=10
total_timesteps=50000  # ~30分钟
learning_rate=1e-4, gamma=0.99, gae_lambda=0.95
clip_range=0.2, ent_coef=0.005, vf_coef=0.5
max_grad_norm=0.5, target_kl=0.02
policy: pi=[128,128], vf=[128,128], Tanh
device=cpu, seed=42
```

## 5. 训练历史

| 运行 | 步数 | 初始距离 | 初速度 | 最高 ep_rew | damage | 结论 |
|------|------|---------|--------|------------|--------|------|
| Run 1-5 | 20-100K | 200-3000 | 50-200 | <0 | 0~稀疏 | 远距+高速→飞过敌机 |
| | | | | | | |
| **Phase 1: 参数调优** | | | | | | |
| Run 8 | 20K | 20(200m) | 5(50m/s) | ~600 | 首次触发 | 距离+速度降低→伤害出现 |
| Run 10 | 50K | 20(200m) | 5(50m/s) | ~1200 | 持续 | 逐步学习接近，但不稳定 |
| | | | | | | |
| **Phase 2: 奖励重设计** | | | | | | |
| Run 11 | 20K | 20(200m) | 5(50m/s) | 1625 | ~16/步 | distance_hold_reward 突破，agent 学会盘旋 |
| **Run 12** | **50K** | **20(200m)** | **5(50m/s)** | **1663** | **~16.4/步** | **✅ 成功：2.8s 击杀，持续盘旋近距** |

### 5.1 Run 12 详细结果（最终成功运行）

| 指标 | 最终值 | 说明 |
|------|--------|------|
| episode_reward | 1663 | 持续上升未衰退 |
| ep_len_mean | 172 步 | ~2.8 秒完成击杀（1000HP / 16.4HP/步 ≈ 61 步理论值） |
| damage_per_step | 16.37 HP | 稳定高伤害，武器持续触发 |
| proximity | 0.5 (饱和) | 始终保持在 300m 内 |
| distance 分量 | ~152 | distance_hold_reward 引导盘旋 |
| heading 分量 | ~72 | 机头持续指向敌机 |

### 5.2 收敛模式（三阶段学习）

```
Phase 1  (0-15K):   探索期 — random exploration, 偶尔触发伤害
Phase 2  (15K-35K): 学习盘旋 — distance_hold_reward 引导 agent 学会减速保持距离
Phase 3  (35K-50K): 高效击杀 — damage/speed/heading 完美平衡，伤害饱和
```

### 5.3 诊断确认

| 诊断 | 结论 |
|------|------|
| 武器伤害验证 | 武器系统有效：每步固定 10 HP 伤害，攻击范围远超 200m |
| 问题根因 | **100% RL 策略层**：初始距离 + 速度 + reward 设计 |
| GPU 加速 | 不实用 — TCP 瓶颈(~15ms/步) 主导，小 MLP 计算开销可忽略 |



## 6. 核心问题（Phase 4 之前，已解决 ✅）

### 6.1 武器系统 — 已确认正常
武器自动开火，攻击范围远超 200m，固定伤害 per step。武器不是瓶颈。

### 6.2 盘旋/缠斗引导 — 已解决
distance_hold_reward 替代 approach_reward，引导 agent 保持 ~200m 距离盘旋。heading_reward 近距离弱化，防止"冲过"行为。

### 6.3 初始距离和速度 — 已优化
初始距离 20 单位（200m），初速度 5 单位（50m/s）。agent 从第一帧就进入武器射程。

### 6.4 速度引导 — 已调整
speed_reward 峰值在 5-20 单位（50-200m/s），鼓励低速盘旋缠斗。

## 7. 当前阶段：移动靶（Simple）

### 7.1 目标
将敌方从 Fixed Target 升级为 Simple 移动靶（无重力，有速度衰减），训练真正空战缠斗。

### 7.2 已完成的改动
- `initialize.py`: 敌方初速度从 `[0,0,0]` → `[5,3,0]`（前50m/s + 侧30m/s）
- `diagnose_simple_vs_simple.py`: 双连接诊断脚本（己方 `port`，敌方 `port+1`，双端口模式）
- `reward.py`: 全面重写面向移动靶 — heading 永不衰减、desertion_penalty 防逃脱、speed 收窄到 50-90m/s

### 7.3 关键发现：双端口协议
Simple vs Simple 服务器为双方分配不同端口：己方连接 `config['port']` (1000)，敌方连接 `port + 1` (1001)。每个客户端需独立的 TCP 连接和 InitData (不同 unit ID)。

### 7.4 下一步
```bash
# 1. UE5 创建 Simple vs Simple 房间（端口 1000）
# 2. 运行双端口诊断验证协议
cd "D:\study\无人系统设计\2026课程大作业\Python"
D:/Anaconda/envs/uav_rl/python.exe diagnose_simple_vs_simple.py
# 3. 诊断通过后 → 改造 train_env.py 支持双连接训练
# 4. 移动靶 reward 测试训练
# 5. evaluate.py 评估移动靶表现
```

## 8. 未来阶段

- **Phase 5**：移动靶训练 — **进行中**（双端口协议已验证，train_env 改造待完成）
- **Phase 6**：在线部署 —— 导出模型用于实时对战，测试泛化

## 9. 运行信息

| 项目 | 值 |
|------|-----|
| Python 环境 | `D:/Anaconda/envs/uav_rl/python.exe` (conda: uav_rl) |
| 关键依赖 | stable-baselines3==2.7.1, gymnasium==1.1.1, numpy, PyTorch, pyyaml |
| Docker | battle_server 端口 8887（管理）, 1000-2000（战斗，Simple vs Simple 用 1000+1001） |
| UE5 | StudentClient, Simple/Fixed 靶机场景 |
| 工作目录 | `D:\study\无人系统设计\2026课程大作业\Python\` |
| 训练输出 | `./output/run_N/`（model/, logs/） |

## 10. 启动流程

### 10.1 Fixed Target 训练/评估（Phase 4 已完成）
1. 确保 Docker battle_server 运行中
2. UE5 StudentClient 创建房间（端口 1000, Simple vs **Fixed Target**）
3. 训练：
```bash
cd "D:\study\无人系统设计\2026课程大作业\Python"
D:/Anaconda/envs/uav_rl/python.exe main.py
```
4. 评估：
```bash
D:/Anaconda/envs/uav_rl/python.exe evaluate.py
```

### 10.2 Simple vs Simple 诊断（Phase 5 进行中，双端口）
1. UE5 创建房间（端口 1000, **Simple vs Simple**）
2. 双端口诊断（己方 1000, 敌方 1001）：
```bash
D:/Anaconda/envs/uav_rl/python.exe diagnose_simple_vs_simple.py
```

## 11. 关键注意事项

- 每次训练前必须**重新创建房间**（上次训练结束房间自动销毁）
- `ConnectionResetError` = 房间失效，需在 UE5 重新创建
- 血量已归一化到 [0,1]，reward 中乘 1000 恢复为实际伤害量
- 无显式开火指令，武器系统自动判断弹道命中
- 每次启动新对话需检查规则 rules 加载、MCP 服务器可用性
