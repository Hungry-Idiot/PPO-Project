import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium
from gymnasium import spaces
from gymnasium.utils import env_checker
import numpy as np
from utils import adaptor, action, observation, reward, truncate, initialize
import yaml


def float_to_bool(f):
    if f == 0.0:
        return False
    elif f == 1.0:
        return True
    else:
        return None


def pack_initial(my_init, enemy_init, unit_id):
    # 转换为整数格式
    integer_my = my_init.astype(np.int32)
    integer_enemy = enemy_init.astype(np.int32)

    room = 114514
    initial_packet = np.array([room, unit_id], dtype=np.int32)

    # 依次拼接视角本体数据和对手数据
    initial_packet = np.append(initial_packet, integer_my)
    initial_packet = np.append(initial_packet, integer_enemy)

    return initial_packet


def split_observation(observation):
    my_state = observation[0:13].astype(np.float64).copy()
    enemy_state = observation[13:26].astype(np.float64).copy()
    terminated = float_to_bool(observation[26])
    return my_state, enemy_state, terminated


def pack_action(action, truncated):
    if truncated:
        truncation = 1.0
    else:
        truncation = 0.0
    full_pack = np.append(action, truncation)
    return full_pack


class TrainEnv(gymnasium.Env):
    def __init__(self, config_path):
        super().__init__()

        # Agent space bounds
        action_upper_bound = np.ones(shape=[4], dtype=np.float64)
        action_lower_bound = np.negative(np.ones(shape=[4], dtype=np.float64))
        self.action_space = spaces.Box(
            shape=[4],
            dtype=np.float64,
            low=action_lower_bound,
            high=action_upper_bound
        )

        observation_upper_bound = np.ones(shape=[21], dtype=np.float64)
        observation_lower_bound = np.negative(np.ones(shape=[21], dtype=np.float64))
        self.observation_space = spaces.Box(
            shape=[21],
            dtype=np.float64,
            low=observation_lower_bound,
            high=observation_upper_bound
        )

        # Initialize adaptor
        self.adaptor = adaptor.NetworkAdaptor(config_path)
        self.adaptor.connect()

        self.adaptor_enemy = adaptor.NetworkAdaptor(config_path)
        self.adaptor_enemy.port += 1  # 强制顺延到 1001 端口
        self.adaptor_enemy.connect()

        self.first_reset = True
        self.episode_id = 0

        self.my_state, self.enemy_state = None, None
        self.state = None

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.save_path = self.config["save_path"]

    def step(self, agent_action):
        # 先根据当前旧状态检查一次是否已经飞出边界
        truncated = truncate.check_truncation(self.my_state, self.enemy_state)

        # Marshal agent actions into real actions and send
        real_action = action.marshal_action(agent_action)
        send_pack = pack_action(real_action, truncated)

        # 构造敌方的零动作并发送
        enemy_action = np.zeros(4, dtype=np.float64)
        enemy_send_pack = pack_action(enemy_action, truncated)

        # 双路发送 CtrlData
        self.adaptor.send_action_packet(send_pack)
        self.adaptor_enemy.send_action_packet(enemy_send_pack)

        # Save previous state for reward calculation
        prev_my_state = self.my_state.copy()
        prev_enemy_state = self.enemy_state.copy()

        # 双路接收 BattleData，敌方视角数据不用，但必须接收以清空 TCP 缓冲区
        original_observation = self.adaptor.get_observation_packet()
        _ = self.adaptor_enemy.get_observation_packet()

        self.my_state, self.enemy_state, terminated = split_observation(original_observation)

        # Process whole state into agent state
        self.state = observation.marshal_observation(self.my_state, self.enemy_state)

        # 接收新状态后再检查一次是否飞出边界
        # 这样可以处理“本 step 动作导致飞出 1500m”的情况
        new_truncated = truncate.check_truncation(self.my_state, self.enemy_state)
        truncated = truncated or new_truncated

        # Gymnasium 中 terminated 表示自然结束，比如击毁/死亡；
        # truncated 表示人为截断，比如飞太远。
        # 如果是飞远截断，就不要同时算作 terminated。
        if truncated:
            terminated = False

        comps = reward.reward_components(
            prev_my_state,
            prev_enemy_state,
            self.my_state,
            self.enemy_state
        )

        # 飞出边界时，额外给一次截断惩罚。
        # 目的：不要让 agent 学到“先打一点伤害，然后高速飞离战场”。
        if truncated:
            remaining_enemy_hp = max(0.0, float(self.enemy_state[12]))
            comps["truncation_penalty"] = -3000.0 - 3000.0 * remaining_enemy_hp
            comps["total"] += comps["truncation_penalty"]
        else:
            comps["truncation_penalty"] = 0.0

        step_reward = comps["total"]

        info = {
            "reward_comps": comps,
        }
        for k, v in comps.items():
            if k != "total":
                info[f"r/{k}"] = float(v)

        return self.state, step_reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 第一次 reset 使用 __init__ 中已经建立好的连接；
        # 后续每个 episode reset 都重新连接 1000/1001
        if self.first_reset:
            self.first_reset = False
        else:
            self.episode_id += 1
            if self.episode_id % 10 == 0:
                print(f"[RESET] reconnect for episode {self.episode_id}", flush=True)

            self.adaptor.reconnect()
            self.adaptor_enemy.reconnect()

        # 生成初始状态并拆分
        init_state = initialize.generate_initial_state()
        my_init = init_state[0:12]
        enemy_init = init_state[12:24]

        # 分别打包，注意敌方视角的 my_init 和 enemy_init 是反过来的
        pack_my = pack_initial(my_init, enemy_init, 1919810)
        pack_enemy = pack_initial(enemy_init, my_init, 1919811)

        # 双路发送 InitData
        self.adaptor.send_initial_packet(pack_my)
        self.adaptor_enemy.send_initial_packet(pack_enemy)

        # 关键：Simple vs Simple 需要双方先发 CtrlData 才会返回 BattleData
        zero_ctrl = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.adaptor.send_action_packet(zero_ctrl)
        self.adaptor_enemy.send_action_packet(zero_ctrl)

        # 必须双路接收 BattleData，敌方视角的数据不用，但必须接收以清空 TCP 缓冲区
        original_observation = self.adaptor.get_observation_packet()
        _ = self.adaptor_enemy.get_observation_packet()

        # 仅使用己方视角的数据进行智能体状态处理
        self.my_state, self.enemy_state, termination = split_observation(original_observation)
        self.state = observation.marshal_observation(self.my_state, self.enemy_state)

        return self.state, {}


if __name__ == '__main__':
    env = TrainEnv('../config/envs.yaml')
    env_checker.check_env(env)