import math
import numpy as np
import torch
import random
import sys

sys.path.append("../../../gym-duckietown/")

from gym_duckietown.envs import DuckietownEnv
from config import buffer_limit, bias, MAX_STEPS, epsilon_decay, device

"""
Seed, ReplayBuffer, and eval fn
Code based on:
https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
"""

# Launch env
def launch_env(MAP_NAME, SEED):
    env = DuckietownEnv(
        map_name=MAP_NAME,
        domain_rand=False,
        draw_bbox=False,
        max_steps=MAX_STEPS,
        seed=SEED,
    )
    print("Initialized environment {}".format(MAP_NAME))
    return env


def seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class ReplayBuffer(object):
    def __init__(self, max_size=buffer_limit):
        self.storage = []
        self.max_size = max_size

    # Expects tuples of (state, next_state, action, reward, done)
    def add(self, state, next_state, action, action_idx, reward, done, add):
        if len(self.storage) < self.max_size:
            self.storage.append(
                (state, next_state, action, action_idx, reward, done, add)
            )
        else:
            # Remove random element in the memory beforea adding a new one
            self.storage.pop(random.randrange(len(self.storage)))
            self.storage.append(
                (state, next_state, action, action_idx, reward, done, add)
            )

    def sample(self, batch_size=100, flat=False):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, next_states, actions, actions_idx, rewards, dones, adds = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for i in ind:
            state, next_state, action, action_idx, reward, done, add = self.storage[i]

            if flat:
                states.append(np.array(state, copy=False).flatten())
                next_states.append(np.array(next_state, copy=False).flatten())
            else:
                states.append(np.array(state, copy=False))
                next_states.append(np.array(next_state, copy=False))
            actions.append(np.array(action, copy=False))
            actions_idx.append(np.array(action_idx, copy=False))
            rewards.append(np.array(reward, copy=False))
            dones.append(done)
            adds.append(add)

        return {
            "states": torch.from_numpy(np.stack(states)).float().to(device),
            "next_states": torch.from_numpy(np.stack(next_states)).float().to(device),
            "actions": torch.from_numpy(np.stack(actions)).float().to(device),
            "actions_idx": torch.from_numpy(np.stack(actions_idx)).to(device),
            "rewards": torch.from_numpy(np.stack(rewards).reshape(-1, 1))
            .float()
            .to(device),
            "dones": torch.tensor(dones),
            "adds": torch.from_numpy(np.stack(adds)).to(device),
        }

    def __len__(self):
        return len(self.storage)


def compute_epsilon(episode, bias=bias, min_epsilon=0.001, max_epsilon=1.0):
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * math.exp(
        -1.0 * (episode + bias) / epsilon_decay
    )
    return epsilon
