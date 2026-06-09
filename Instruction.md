课程大作业 —— PPO 强化学习空战智能体训练
========================

> Last updated: 2026-06-09  
> 当前状态：**Simple vs Simple 移动靶阶段已完成**；下一阶段目标：**挑战 Junior 机型**。

* * *

1. ## 项目目标

-------

使用 **PPO（Proximal Policy Optimization, Stable-Baselines3）** 训练战斗机智能体，在 UE5 仿真环境中击败敌方战机。

当前采用渐进式路线：
    Fixed Target
      → Simple vs Simple 移动靶
      → Junior 机型
      → 在线对战/最终提交

截至目前：

* Fixed Target 阶段已完成。

* Simple vs Simple 移动靶阶段已完成，当前最佳模型可在约 **100 step** 内稳定击杀敌机。

* 下一阶段开始探索 **Junior**，重点处理重力、初速度、高度保持、油门上限和坠毁风险。

* * *

2. ## 系统架构

-------

    Python Agent (PPO/SB3)
      └─ TCP Socket
           ├─ 己方端口: 127.0.0.1:1000
           └─ 敌方端口: 127.0.0.1:1001   # Simple vs Simple / 双智能体房间
                └─ Docker battle_server
                     ├─ 管理端口: 8887
                     └─ 战斗端口: 1000-2000
                          └─ UE5 StudentClient

* **Docker battle_server**：管理房间和战斗生命周期。

* **UE5 StudentClient**：提供可视化渲染和物理仿真。

* **Python Agent**：通过 TCP Socket 每步发送控制、接收观测。

* **Simple vs Simple 模式**：需要同时连接两个端口，己方使用 `port`，敌方使用 `port + 1`。

推荐启动顺序：
    Docker battle_server
      → UE5 StudentClient 创建房间并 Start
      → Python 训练/评估脚本

* * *

3. ## 通信协议

-------

### 3.1 初始包 InitData（400 bytes）

    INITIAL_PACKET_FORMAT = "<26i296x"
    # [room(1), unit(1), my_state(12), enemy_state(12)] → 26 int32

双端口 Simple vs Simple 中需要分别发送两份 InitData：
    己方连接 1000: unit_id = 1919810, my=己方, enemy=敌方
    敌方连接 1001: unit_id = 1919811, my=敌方, enemy=己方

双方初始 12 维状态：
    位置3 + 姿态3 + 线速度3 + 角速度3

### 3.2 观测包 BattleData（216 bytes, 27 doubles）

    GETTING_PACKET_FORMAT = "=27d"
    # [my_state(13), enemy_state(13), is_done(1)]

每架飞机 13 维：
    位置3 + 姿态3 + 线速度3 + 角速度3 + 血量HP

注意：

* HP 归一化到 `[0, 1]`。

* `is_done: 0.0 = 继续, 1.0 = 终止`。

* 距离单位为 **10m**，例如 `20 = 200m`。

* 角度单位为 **弧度**。

### 3.3 控制包 CtrlData（40 bytes, 5 doubles）

    SENDING_PACKET_FORMAT = "<5d"
    # [throttle, pitch, roll, yaw, truncation]

控制含义：
    throttle:   [0, 1]
    pitch:      [-1, 1]
    roll:       [-1, 1]
    yaw:        [-1, 1]
    truncation: 0.0 / 1.0

武器系统无显式开火指令。攻击范围与敌方受击体积重叠时，仿真平台自动造成固定伤害。

* * *

4. ## Python 代码结构

--------------

    Python/
    ├── main.py                         # 从零训练入口
    ├── main_finetune.py                # 从已有 checkpoint 继续微调
    ├── evaluate.py                     # 单模型详细评估，打印逐 step 信息
    ├── evaluate_checkpoints_py39.py    # 批量 checkpoint 评估，兼容 Python 3.9
    ├── config/envs.yaml                # host/port/save_path 等配置
    ├── envs/train_env.py               # Gymnasium Env，双端口训练核心
    ├── utils/
    │   ├── adaptor.py                  # TCP 网络适配器，含 timeout/reconnect/sendall
    │   ├── action.py                   # 动作映射；Simple 阶段限制油门上限
    │   ├── observation.py              # 21 维观测处理
    │   ├── reward.py                   # 奖励函数
    │   ├── initialize.py               # 初始状态生成
    │   ├── truncate.py                 # 飞远截断检查
    │   └── callback.py                 # 奖励分量 CSV 日志
    ├── diagnose_env.py
    ├── diagnose_combat.py
    ├── diagnose_damage_short.py
    ├── diagnose_weapon_range.py
    └── diagnose_simple_vs_simple.py    # Simple vs Simple 双端口诊断

* * *

5. ## 观测空间与动作空间

------------

### 5.1 观测空间（21 维）

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

### 5.2 动作空间（4 维）

    action = [throttle, pitch, roll, yaw]  # 神经网络输出全部 ∈ [-1, 1]

Simple 移动靶最终成功版本中，`utils/action.py` 对油门进行了限制：
    raw_throttle = (clipped_action[0] + 1.0) / 2.0
    processed_action[0] = 0.4 * raw_throttle

这样物理油门范围从 `[0, 1]` 限制到 `[0, 0.4]`，用于抑制 Simple 机体高速冲出战场。

> Junior 阶段注意：Junior 受重力影响，`0.4` 油门上限可能不足，进入 Junior 前需要重新验证油门、初速度和高度保持。

* * *

6. ## 当前 Simple vs Simple 成功版本

---------------------------

### 6.1 初始状态

    my:    pos=(0, 0, 1000),  rot=(0,0,0), vel=(5,0,0)
    enemy: pos=(20,0,1000),  rot=(0,0,0), vel=(5,3,0)

说明：

* 初始距离 `20`，即 200m。

* 己方初始前向速度 `5`，即 50m/s。

* 敌方 Simple 移动靶带有侧向速度，用于测试追瞄稳定性。

### 6.2 双端口训练逻辑

`TrainEnv` 当前关键逻辑：

1. `__init__` 中连接己方 `port` 与敌方 `port + 1`。

2. 每个 episode 后续 reset 都执行双路 `reconnect()`。

3. reset 时分别向两路发送 InitData。

4. InitData 后，双方都先发送一次零 CtrlData，触发仿真平台返回首个 BattleData。

5. step 时，己方发送 PPO 动作，敌方发送零动作。

6. 两路都接收 BattleData，其中训练只使用己方视角观测。

### 6.3 截断与飞远惩罚

`utils/truncate.py`：
    if dist > 150.0:  # 150单位 = 1500m
        return True

`envs/train_env.py` 中对飞远截断额外惩罚：
    remaining_enemy_hp = max(0.0, float(self.enemy_state[12]))
    comps["truncation_penalty"] = -3000.0 - 3000.0 * remaining_enemy_hp
    comps["total"] += comps["truncation_penalty"]

目的：禁止模型学习“先蹭伤害，然后高速飞离战场”的局部最优策略。

### 6.4 奖励函数关键点

Simple 成功版本的奖励设计重点：

* 持续伤害奖励作为主要正反馈。

* `kill_bonus` 提升到较大值，用于明确鼓励最终击杀。

* `death_penalty` 采用 transition-only，避免死亡后重复扣分。

* `desertion_penalty` 与 `truncation_penalty` 防止飞离。

* `speed_reward` 鼓励较低速度的稳定追瞄。

* `proximity_reward` 防止过近贴脸/碰撞。

* `heading_reward` 鼓励机头朝向敌机。

* * *

7. ## 训练与评估历史

----------

### 7.1 Fixed Target 阶段

| 阶段              | 结论          |
| --------------- | ----------- |
| 初期远距/高速训练       | 容易飞过敌机，伤害稀疏 |
| 降低初始距离和速度       | 开始稳定触发武器伤害  |
| reward 重设计      | 学会近距盘旋和持续伤害 |
| Fixed Target 最终 | 可在数秒内击杀固定靶  |

### 7.2 Simple vs Simple 阶段关键里程碑

| 阶段                | 现象                                           | 处理                                 |
| ----------------- | -------------------------------------------- | ---------------------------------- |
| 双端口诊断             | Simple vs Simple 需要 1000/1001 双连接            | 编写 `diagnose_simple_vs_simple.py`  |
| 训练初期卡住            | reset 后旧 socket 不再返回 BattleData              | 每个 episode 后双路 `reconnect()`       |
| 1K/10K 初训         | 能训练，但 reward 被速度/死亡惩罚干扰                      | 调整 reward 分量，death transition-only |
| 50K 训练            | 训练 reward 上升，但 deterministic evaluation 不能击杀 | 批量评估 checkpoint                    |
| 原始 45K checkpoint | 能打到敌方约 13% HP，但随后飞离到 30km                    | 从 45K 开始微调                         |
| 第一次微调             | 飞远惩罚过弱，仍未击杀                                  | 强化截断惩罚、增大 kill bonus、限制油门          |
| 第二次微调             | 出现稳定击杀                                       | 选择 60K 微调模型                        |

### 7.3 当前最佳 Simple 模型

当前推荐模型：
    finetune_model_60000_steps.zip

建议另存为：
    ./output/best_simple_model.zip

`evaluate.py` 连续 5 次复测结果一致：
    Steps:        100
    Total DMG:    990.0 HP
    Enemy HP:     0.0000
    Self HP:      0.3200
    KILL:         YES
    Min distance: 27.2 m

典型轨迹：

* 前 99 步每步稳定造成 10 HP。

* 距离从约 200m 缩短到最低约 27m。

* 击杀时距离约 69m。

* 不再出现飞离 30km 的问题。

* * *

8. ## 当前运行命令

---------

### 8.1 训练/微调

从最佳原始 checkpoint 继续微调：
    cd "D:\study\无人系统设计\2026课程大作业\Python"
    D:/Anaconda/envs/uav_rl/python.exe main_finetune.py --model ./output/run_11/model/model_45000_steps.zip --timesteps 30000 --lr 3e-5 --ent-coef 0.0

注意：`run_11` 需替换为实际目录。

### 8.2 批量评估 checkpoint

    D:/Anaconda/envs/uav_rl/python.exe evaluate_checkpoints_py39.py --episodes 10 --models ./output/finetune_xxx/model/finetune_model_60000_steps.zip ./output/finetune_xxx/model/finetune_model_70000_steps.zip ./output/finetune_xxx/model/ppo_finetuned.zip

### 8.3 单模型详细评估

    D:/Anaconda/envs/uav_rl/python.exe evaluate.py

`evaluate.py` 中应加载：
    model_path = "./output/best_simple_model.zip" 

* * *

9. ## 下一阶段：Junior 挑战计划

--------------------

### 9.1 Junior 与 Simple 的关键差异

Simple：
    无重力，主要问题是追瞄、速度和飞远。

Junior：
    受重力影响，需要维持高度、速度和升力。

因此，Simple 阶段的策略不能直接照搬到 Junior。尤其要重新检查：

* 油门上限 `0.4` 是否太低。

* 初始高度是否足够。

* 初始速度是否能维持飞行。

* pitch/roll/yaw 控制是否会导致坠毁。

* reward 是否充分惩罚低空和坠毁。

### 9.2 Junior 前置诊断

进入训练前，先做诊断，不要直接 PPO 训练。

建议新增或修改诊断脚本：
    diagnose_junior_dynamics.py

需要测试：

1. 零动作下 Junior 是否快速掉高度。

2. 固定油门 `0.4 / 0.6 / 0.8 / 1.0` 下能否维持高度。

3. 固定 pitch 小角度下是否能爬升或平飞。

4. 控制包范围是否与 Simple 一致。

5. Junior vs Fixed Target 是否能稳定连通、收包、造成伤害。

### 9.3 Junior 初始状态建议

初始版本可先采用更保守配置：
    高度: >= 1000
    速度: 比 Simple 更高，例如 8~12 单位
    距离: 200m 或 300m
    敌方: 先 Fixed Target，再移动靶

推荐路线：
    Junior vs Fixed Target
      → Junior vs Simple / Junior moving target
      → Junior vs Junior

不要一开始就上 Junior vs Junior。

### 9.4 Junior reward 初步方向

在 Simple reward 基础上增加/强化：
    altitude_reward / altitude_penalty:
      低空强惩罚，避免坠毁。

    vertical_speed_penalty:
      惩罚高速下坠。

    speed_reward:
      Junior 可能需要更高速度区间，不能沿用 Simple 的低速偏好。

    throttle_regularization:
      可轻微惩罚长期满油门，但不能限制到无法维持飞行。

    crash/death_penalty:
      继续使用 transition-only，避免重复扣分。

### 9.5 Junior 动作映射注意

Simple 成功版本限制：
    processed_action[0] = 0.4 * raw_throttle

Junior 阶段建议先改成可配置，例如：
    MAX_THROTTLE = 0.8  # Junior 初始测试值，必要时升到 1.0
    processed_action[0] = MAX_THROTTLE * raw_throttle

更好的做法是将 `MAX_THROTTLE` 放入配置文件，避免 Simple 与 Junior 来回手改代码。

* * *

10. ## 运行注意事项

----------

* 每次训练前通常需要重新创建 UE5 房间。

* Simple vs Simple 使用双端口：己方 `1000`，敌方 `1001`。

* 如果出现 `ConnectionResetError`、`Socket closed`、`Timeout waiting for packet`，优先检查房间是否已结束或端口是否被回收。

* `maxEpisodes` 要设置足够大，避免训练中途房间被回收。

* 血量在观测中为 `[0,1]`，统计伤害时通常乘以 1000。

* 评估 checkpoint 时，如果出现 `steps=1, total_damage=0, platform_done=1`，通常是上一局残留终止状态，不应计为真实击杀。

* 无显式开火指令，武器由仿真平台自动判定。

* 进入 Junior 前，先保留 Simple 成功代码和模型，建议打 Git tag 或单独分支。

* * *

11. ## 当前下一步清单

-----------

1. 为 Junior 建立单独实验分支或至少记录清楚改动。

2. 编写 Junior 动力学诊断脚本。

3. 先验证 Junior 平飞/高度保持，再训练 Junior vs Fixed Target。

4. 根据 Junior 诊断结果调整 `action.py` 的油门上限、`initialize.py` 初始速度/高度和 `reward.py` 高度奖励。
