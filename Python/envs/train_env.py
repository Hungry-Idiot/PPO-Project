import gymnasium
from gymnasium import spaces
from gymnasium.utils import env_checker
import numpy as np
from utils import adaptor, action, observation, reward, truncate, initialize
import yaml
import os

def float_to_bool(f):
    if f == 0.0:
        return False
    elif f == 1.0:
        return True
    else:
        return None


def pack_initial(initial_observation):
    integer_observation = initial_observation.astype(np.int32)
    room = 114514
    unit = 1919810
    initial_packet = np.array([room], dtype=np.int32)
    initial_packet = np.append(initial_packet, unit)
    initial_packet = np.append(initial_packet, integer_observation)
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
        self.action_space = spaces.Box(shape=[4], dtype=np.float64, low=action_lower_bound, high=action_upper_bound)

        observation_upper_bound = np.ones(shape=[21], dtype=np.float64)
        observation_lower_bound = np.negative(np.ones(shape=[21], dtype=np.float64))
        self.observation_space = spaces.Box(shape=[21], dtype=np.float64, low=observation_lower_bound, high=observation_upper_bound)

        # Initialize adaptor
        self.adaptor = adaptor.NetworkAdaptor(config_path)
        self.adaptor.connect()

        self.my_state, self.enemy_state = None, None
        self.state = None
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.save_path = self.config["save_path"]


    def step(self, agent_action):
        # Check for truncation first
        truncated = truncate.check_truncation(self.my_state, self.enemy_state)

        # Marshal agent actions into real actions and send
        # First marshal unified actions into platform actions
        real_action = action.marshal_action(agent_action)
        # Then append truncation flag to the packet and send
        send_pack = pack_action(real_action, truncated)
        self.adaptor.send_action_packet(send_pack)

        # Save previous state for reward calculation
        prev_my_state = self.my_state.copy()
        prev_enemy_state = self.enemy_state.copy()

        # Get new observations and unmarshal
        original_observation = self.adaptor.get_observation_packet()
        self.my_state, self.enemy_state, terminated = split_observation(original_observation)
        # Process whole state into agent state
        self.state = observation.marshal_observation(self.my_state, self.enemy_state)

        # Check for termination
        # But truncation has a higher priority
        if truncated:
            terminated = False

        comps = reward.reward_components(prev_my_state, prev_enemy_state, self.my_state, self.enemy_state)
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
        self.adaptor.reconnect()
        new_initial_packet = pack_initial(initialize.generate_initial_state())
        self.adaptor.send_initial_packet(new_initial_packet)
        # Get new observations and unmarshal
        original_observation = self.adaptor.get_observation_packet()
        self.my_state, self.enemy_state, termination = split_observation(original_observation)
        # Process whole state into agent state
        self.state = observation.marshal_observation(self.my_state, self.enemy_state)
        return self.state, {}


if __name__ == '__main__':
    env = TrainEnv('../config/envs.yaml')
    env_checker.check_env(env)