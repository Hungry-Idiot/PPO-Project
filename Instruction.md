# 课程大作业 —— PPO 强化学习空战智能体训练

## 1. 项目目标

使用 **PPO（Proximal Policy Optimization, Stable-Baselines3）** 训练战斗机智能体，在 UE5 仿真中**击败固定靶机（Fixed Target Drone）**。

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
└── diagnose_weapon_range.py  # 武器攻击范围测量（单连接二分搜索）
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

### 4.3 奖励函数（reward.py — 当前版本）
```python
damage_reward:   damage_dealt * 2.0 - damage_taken * 1.5
heading_reward:  heading_dot * 2.0        # 机头点积×2，无阈值
approach_reward: (prev_dist - curr_dist) * 2.0
speed_reward:    <50=-1.0, 50-100=线性, >=100=0.2
step_penalty:    -0.02
altitude_penalty: -10.0 if z < 50 else 0.0
```

### 4.4 初始状态（initialize.py — 当前版本）
```python
my:    pos=(0,0,1000), forward_vel=50    # 500m/s
enemy: pos=(200,0,1000), vel=0          # 初始相距 2000m，固定靶
```

### 4.5 PPO 超参（main.py — 当前版本）
```python
n_steps=512, batch_size=128, n_epochs=10
total_timesteps=20000  # ~11分钟
learning_rate=1e-4, gamma=0.99, gae_lambda=0.95
clip_range=0.2, ent_coef=0.005, vf_coef=0.5
max_grad_norm=0.5, target_kl=0.02
policy: pi=[128,128], vf=[128,128], Tanh
device=cpu, seed=42
```

## 5. 训练历史

| 运行 | 步数 | 初始距离 | 初速度 | 最高 ep_rew | damage | 结论 |
|------|------|---------|--------|------------|--------|------|
| Run 1 | 100K | 3000 | 200 | -132 | 未知 | 收敛但模型未保存 |
| Run 2 | 100K | 3000 | 200 | -12,500 | 未知 | 发散 |
| Run 3 | 50K | 3000 | 200 | ~884→-68 | 0 | 太远(30km)，飞不到 |
| Run 4 | 50K | 500 | 200 | ~-3400 | 0 | 速度太快(2000m/s)，一帧冲过 |
| Run 5 | 20K | 200 | 50 | ~-3000 | 稀疏(3/40) | 接近敌机但伤害极稀疏 |

**Run 5 亮点**：iter 19 出现 `approach=1.44, heading=0.805`，证明 agent 有能力学会接近，但立刻冲过敌机。伤害仅在 3/40 个 iteration 中非零。

| 诊断 | 日期 | 方法 | 结论 |
|------|------|------|------|
| 武器伤害验证 | 2026-06-08 | init_dist=3(30m), vel=0.5, 零操作, 单连接 | **武器系统有效**：每步固定 10 HP 伤害，100 步击杀 |
| 武器范围测量 | 2026-06-08 | 单连接二分搜索, 距离 1~300 单位 | 攻击范围远超 200m（300 单位=3000m 仍触发伤害），无角度/横向偏移限制 |
| 矛盾数据分析 | 2026-06-08 | 多连接 vs 单连接对比 | 频繁断连重连（每个距离测一次）导致 TCP 连接污染，InitData 部分失效——不是武器问题，是诊断方法问题 |
| 训练真实根因 | 2026-06-08 | 综合分析 | 武器有效、射程足够。问题 100% 在 RL 策略层：初始距离太远(2000m)+ 速度太快(500m/s)→ 冲过窗口，无盘旋机制。

## 6. 核心问题（已确认，待修复）

### 6.1 武器系统已验证正常
武器自动开火，控制协议无显式开火字段。诊断确认：**攻击范围远超 200m（3000m 仍触发），固定 10 HP/步伤害，无角度/偏移限制。** 武器不是问题。

### 6.2 无盘旋/缠斗引导（ROOT CAUSE）
heading 和 approach 奖励引导 agent 接近敌机，但 agent 接近后立即冲过 —— 没有机制让它**减速盘旋**或**反复进入射程**。

### 6.3 初始距离和速度不匹配
初始距离 200 单位（2000m）、初速度 50（500m/s）。agent ~4 步飞过敌机，来不及做出有意义机动。武器射程虽远超 200m，但 agent 缺乏"停留"动机。

### 6.4 时间尺度过短
1 步 = 1/60s，512 步 ≈ 8.5 秒。agent 飞过敌机后没有足够时间掉头再次尝试。

## 7. 建议方案（按优先级，已更新）

### 7.1 降低初始距离 + 速度（最优先）
- initialize.py：初始距离 200 → **20**（200m，出生即在武器射程内）
- initialize.py：初速度 50 → **10**（100m/s，足够慢以留在射程内）
- 目的：让 agent 从第一帧就触发 damage reward，不再需要"发现"伤害

### 7.2 近距奖励（proximity reward）
在 reward.py 中添加 proximity_reward（距离 < X 时持续给正奖励），引导 agent 盘旋而非冲过。

### 7.3 速度引导调整
speed_reward 改为奖励低速（目标 10-20 而非 50-100），让 agent 学会减速盘旋。

### 7.4 增加 episode 长度
n_steps：512 → 1024，给 agent 更多时间完成击杀或掉头。

### 7.5 课程学习（备选）
Stage 1: 仅 heading + approach + proximity + damage；Stage 2: 增加距离，完整 reward。

## 8. 未来阶段

- **Phase 5**：模型评估 —— 加载模型，仿真运行，分析 episode reward、伤害量、击杀率
- **Phase 6**：在线部署 —— 导出模型用于实时对战，测试泛化

## 9. 运行信息

| 项目 | 值 |
|------|-----|
| Python 环境 | `D:/Anaconda/envs/uav_rl/python.exe` (conda: uav_rl) |
| 关键依赖 | stable-baselines3==2.7.1, gymnasium==1.1.1, numpy, PyTorch, pyyaml |
| Docker | battle_server 端口 8887（管理）, 1000-2000（战斗） |
| UE5 | StudentClient, Fixed 靶机场景 |
| 工作目录 | `D:\study\无人系统设计\2026课程大作业\Python\` |
| 训练输出 | `./output/run_N/`（model/, logs/） |

## 10. 启动流程

1. 确保 Docker battle_server 运行中
2. UE5 StudentClient 创建房间（端口 1000, Fixed 靶机）
3. 执行训练：
```bash
cd "D:\study\无人系统设计\2026课程大作业\Python"
D:/Anaconda/envs/uav_rl/python.exe main.py
```
4. 输出：`./output/run_N/` → `model/`（检查点）, `logs/`（TensorBoard + reward_components.csv）

## 11. 关键注意事项

- 每次训练前必须**重新创建房间**（上次训练结束房间自动销毁）
- `ConnectionResetError` = 房间失效，需在 UE5 重新创建
- 血量已归一化到 [0,1]，reward 中乘 1000 恢复为实际伤害量
- 无显式开火指令，武器系统自动判断弹道命中
- 每次启动新对话需检查规则 rules 加载、MCP 服务器可用性
