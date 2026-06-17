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
    integer_my = my_init.astype(np.int32)
    integer_enemy = enemy_init.astype(np.int32)

    room = 114514
    initial_packet = np.array([room, unit_id], dtype=np.int32)
    initial_packet = np.append(initial_packet, integer_my)
    initial_packet = np.append(initial_packet, integer_enemy)

    return initial_packet


def split_observation(observation):
    my_state = observation[0:13].astype(np.float64).copy()
    enemy_state = observation[13:26].astype(np.float64).copy()
    terminated = float_to_bool(observation[26])
    return my_state, enemy_state, terminated


def pack_action(real_action, is_done=False):
    done_flag = 1.0 if is_done else 0.0
    return np.append(real_action, done_flag).astype(np.float64)


class TrainEnv(gymnasium.Env):
    def __init__(self, config_path):
        super().__init__()

        action_upper_bound = np.ones(shape=[4], dtype=np.float64)
        action_lower_bound = -np.ones(shape=[4], dtype=np.float64)
        self.action_space = spaces.Box(
            shape=[4],
            dtype=np.float64,
            low=action_lower_bound,
            high=action_upper_bound
        )

        observation_upper_bound = np.ones(shape=[21], dtype=np.float64)
        observation_lower_bound = -np.ones(shape=[21], dtype=np.float64)
        self.observation_space = spaces.Box(
            shape=[21],
            dtype=np.float64,
            low=observation_lower_bound,
            high=observation_upper_bound
        )

        self.adaptor = adaptor.NetworkAdaptor(config_path)
        self.adaptor.connect()

        self.first_reset = True
        self.episode_id = 0

        self.my_state, self.enemy_state = None, None
        self.state = None

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.save_path = self.config["save_path"]

    def step(self, agent_action):
        # Junior 阶段：本地安全检查只作为惩罚信号，不作为 done 信号发给平台。
        was_locally_unsafe = truncate.check_truncation(self.my_state, self.enemy_state)

        real_action = action.marshal_action(agent_action)

        # 不再把 local truncation 写入 CtrlData.is_done。
        # 平台 episode 结束只以服务器 BattleData.m_is_done 为准。
        send_pack = pack_action(real_action, is_done=False)
        self.adaptor.send_action_packet(send_pack)

        prev_my_state = self.my_state.copy()
        prev_enemy_state = self.enemy_state.copy()

        original_observation = self.adaptor.get_observation_packet()
        self.my_state, self.enemy_state, terminated = split_observation(original_observation)

        if terminated is None:
            terminated = False

        self.state = observation.marshal_observation(self.my_state, self.enemy_state)

        is_locally_unsafe = truncate.check_truncation(self.my_state, self.enemy_state)
        local_truncated = bool(was_locally_unsafe or is_locally_unsafe)

        comps = reward.reward_components(
            prev_my_state,
            prev_enemy_state,
            self.my_state,
            self.enemy_state
        )

        # local_truncated 只惩罚，不结束 episode。
        if local_truncated:
            remaining_enemy_hp = max(0.0, float(self.enemy_state[12]))
            comps["truncation_penalty"] = -800.0 - 1200.0 * remaining_enemy_hp
            comps["total"] += comps["truncation_penalty"]
        else:
            comps["truncation_penalty"] = 0.0

        step_reward = comps["total"]

        info = {
            "reward_comps": comps,
            "local_truncated": local_truncated,
        }
        for k, v in comps.items():
            if k != "total":
                info[f"r/{k}"] = float(v)

        # 注意：truncated 固定返回 False，避免 SB3 提前 reset 导致平台不同步。
        return self.state, step_reward, bool(terminated), False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.first_reset:
            self.first_reset = False
        else:
            self.episode_id += 1
            if self.episode_id % 10 == 0:
                print(f"[RESET] reconnect for episode {self.episode_id}", flush=True)

            self.adaptor.reconnect()

        init_state = initialize.generate_initial_state()
        my_init = init_state[0:12]
        enemy_init = init_state[12:24]

        pack_my = pack_initial(my_init, enemy_init, 1919810)
        self.adaptor.send_initial_packet(pack_my)

        original_observation = self.adaptor.get_observation_packet()
        self.my_state, self.enemy_state, termination = split_observation(original_observation)

        self.state = observation.marshal_observation(self.my_state, self.enemy_state)

        return self.state, {}


if __name__ == "__main__":
    env = TrainEnv("../config/envs.yaml")
    env_checker.check_env(env)