import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium
from gymnasium import spaces
from gymnasium.utils import env_checker
import numpy as np
from utils import adaptor, action, observation, reward, truncate, initialize
import yaml


def get_nested(config, keys, default=None):
    cur = config
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def float_to_bool(f):
    if f == 0.0:
        return False
    elif f == 1.0:
        return True
    else:
        return None


def pack_initial(my_init, enemy_init, room_id, unit_id):
    integer_my = my_init.astype(np.int32)
    integer_enemy = enemy_init.astype(np.int32)

    initial_packet = np.array([room_id, unit_id], dtype=np.int32)
    initial_packet = np.append(initial_packet, integer_my)
    initial_packet = np.append(initial_packet, integer_enemy)

    return initial_packet


def split_observation(observation_packet):
    my_state = observation_packet[0:13].astype(np.float64).copy()
    enemy_state = observation_packet[13:26].astype(np.float64).copy()
    terminated = float_to_bool(observation_packet[26])
    return my_state, enemy_state, terminated


def pack_action(real_action, truncated):
    if truncated:
        truncation = 1.0
    else:
        truncation = 0.0

    full_pack = np.append(real_action, truncation)
    return full_pack.astype(np.float64)


class TrainEnv(gymnasium.Env):
    def __init__(self, config_path):
        super().__init__()

        self.config_path = config_path

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.save_path = self.config.get("save_path", "../logs/training/")

        self.room_id = int(get_nested(self.config, ["room", "room_id"], 114514))
        self.my_unit_id = int(get_nested(self.config, ["room", "my_unit_id"], 1919810))
        self.enemy_unit_id = int(get_nested(self.config, ["room", "enemy_unit_id"], 1919811))

        # 重要：
        # 旧的 envs.yaml 没有 room.dual_port 字段。
        # 为了不破坏 Simple vs Simple 成功版本，默认 dual_port=True。
        # Junior 专用 envs_junior.yaml 中显式设置 dual_port=false。
        self.dual_port = bool(get_nested(self.config, ["room", "dual_port"], True))
        self.enemy_port_offset = int(get_nested(self.config, ["room", "enemy_port_offset"], 1))

        # Agent action space
        action_upper_bound = np.ones(shape=[4], dtype=np.float64)
        action_lower_bound = -np.ones(shape=[4], dtype=np.float64)
        self.action_space = spaces.Box(
            shape=[4],
            dtype=np.float64,
            low=action_lower_bound,
            high=action_upper_bound,
        )

        # Observation space
        observation_upper_bound = np.ones(shape=[21], dtype=np.float64)
        observation_lower_bound = -np.ones(shape=[21], dtype=np.float64)
        self.observation_space = spaces.Box(
            shape=[21],
            dtype=np.float64,
            low=observation_lower_bound,
            high=observation_upper_bound,
        )

        self.adaptor = adaptor.NetworkAdaptor(config_path)
        self.adaptor.connect()

        self.adaptor_enemy = None
        if self.dual_port:
            self.adaptor_enemy = adaptor.NetworkAdaptor(config_path)
            self.adaptor_enemy.port += self.enemy_port_offset
            self.adaptor_enemy.connect()
            print(
                f"[TrainEnv] dual-port mode: my_port={self.adaptor.port}, "
                f"enemy_port={self.adaptor_enemy.port}",
                flush=True,
            )
        else:
            print(
                f"[TrainEnv] single-port mode: my_port={self.adaptor.port}",
                flush=True,
            )

        self.first_reset = True
        self.episode_id = 0

        self.my_state = None
        self.enemy_state = None
        self.state = None

    def _reconnect(self):
        self.adaptor.reconnect()

        if self.dual_port and self.adaptor_enemy is not None:
            self.adaptor_enemy.reconnect()

    def _send_initial_packets(self):
        init_state = initialize.generate_initial_state()
        my_init = init_state[0:12]
        enemy_init = init_state[12:24]

        pack_my = pack_initial(
            my_init=my_init,
            enemy_init=enemy_init,
            room_id=self.room_id,
            unit_id=self.my_unit_id,
        )
        self.adaptor.send_initial_packet(pack_my)

        if self.dual_port and self.adaptor_enemy is not None:
            pack_enemy = pack_initial(
                my_init=enemy_init,
                enemy_init=my_init,
                room_id=self.room_id,
                unit_id=self.enemy_unit_id,
            )
            self.adaptor_enemy.send_initial_packet(pack_enemy)

    def _send_warmup_ctrl_and_recv_initial_observation(self):
        zero_ctrl = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        self.adaptor.send_action_packet(zero_ctrl)

        if self.dual_port and self.adaptor_enemy is not None:
            self.adaptor_enemy.send_action_packet(zero_ctrl)

        original_observation = self.adaptor.get_observation_packet()

        if self.dual_port and self.adaptor_enemy is not None:
            _ = self.adaptor_enemy.get_observation_packet()

        return original_observation

    def step(self, agent_action):
        if self.my_state is None or self.enemy_state is None:
            raise RuntimeError("TrainEnv.step() called before reset().")

        truncated = truncate.check_truncation(self.my_state, self.enemy_state)

        real_action = action.marshal_action(agent_action)
        send_pack = pack_action(real_action, truncated)

        if self.dual_port and self.adaptor_enemy is not None:
            enemy_action = np.zeros(4, dtype=np.float64)
            enemy_send_pack = pack_action(enemy_action, truncated)
        else:
            enemy_send_pack = None

        prev_my_state = self.my_state.copy()
        prev_enemy_state = self.enemy_state.copy()

        self.adaptor.send_action_packet(send_pack)

        if self.dual_port and self.adaptor_enemy is not None:
            self.adaptor_enemy.send_action_packet(enemy_send_pack)

        original_observation = self.adaptor.get_observation_packet()

        if self.dual_port and self.adaptor_enemy is not None:
            _ = self.adaptor_enemy.get_observation_packet()

        self.my_state, self.enemy_state, terminated = split_observation(original_observation)

        if terminated is None:
            terminated = False

        self.state = observation.marshal_observation(self.my_state, self.enemy_state)

        new_truncated = truncate.check_truncation(self.my_state, self.enemy_state)
        truncated = truncated or new_truncated

        if truncated:
            terminated = False

        comps = reward.reward_components(
            prev_my_state,
            prev_enemy_state,
            self.my_state,
            self.enemy_state,
        )

        if truncated:
            remaining_enemy_hp = max(0.0, float(self.enemy_state[12]))
            comps["truncation_penalty"] = -3000.0 - 3000.0 * remaining_enemy_hp
            comps["total"] += comps["truncation_penalty"]
        else:
            comps["truncation_penalty"] = 0.0

        step_reward = comps["total"]

        info = {
            "reward_comps": comps,
            "dual_port": self.dual_port,
        }

        for k, v in comps.items():
            if k != "total":
                info[f"r/{k}"] = float(v)

        return self.state, step_reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.first_reset:
            self.first_reset = False
        else:
            self.episode_id += 1

            if self.episode_id % 10 == 0:
                print(f"[RESET] reconnect for episode {self.episode_id}", flush=True)

            self._reconnect()

        self._send_initial_packets()

        original_observation = self._send_warmup_ctrl_and_recv_initial_observation()

        self.my_state, self.enemy_state, terminated = split_observation(original_observation)
        self.state = observation.marshal_observation(self.my_state, self.enemy_state)

        info = {
            "dual_port": self.dual_port,
            "episode_id": self.episode_id,
        }

        return self.state, info


if __name__ == "__main__":
    env = TrainEnv("../config/envs.yaml")
    env_checker.check_env(env)