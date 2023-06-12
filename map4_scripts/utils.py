import math
import numpy as np
import torch
import random

from collections import deque

from gym_duckietown.envs import DuckietownEnv
from map4_scripts.config import custom_params
from map4_scripts.wrappers3 import NormalizeWrapper, ResizeWrapper, ImgWrapper, ModifiedRewardWrapper
from map4_scripts.config import seeds_dict
from map4_scripts.feature_engineering import get_dist_to_stop
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
        max_steps=custom_params['MAX_STEPS'],
        seed=SEED,
    )
    print("Initialized environment {}".format(MAP_NAME))
    return env


def seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class ReplayBuffer(object):
    def __init__(self, max_size=custom_params['buffer_limit']):
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
            "states": torch.from_numpy(np.stack(states)).float().to(custom_params['device']),
            "next_states": torch.from_numpy(np.stack(next_states)).float().to(custom_params['device']),
            "actions": torch.from_numpy(np.stack(actions)).float().to(custom_params['device']),
            "actions_idx": torch.from_numpy(np.stack(actions_idx)).to(custom_params['device']),
            "rewards": torch.from_numpy(np.stack(rewards).reshape(-1, 1))
            .float()
            .to(custom_params['device']),
            "dones": torch.tensor(dones),
            "adds": torch.from_numpy(np.stack(adds)).to(custom_params['device']),
        }

    def __len__(self):
        return len(self.storage)


def compute_epsilon(episode, bias=custom_params['bias'], min_epsilon=0.001, max_epsilon=1.0):
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * math.exp(
        -1.0 * (episode + bias) / custom_params['epsilon_decay']
    )
    return epsilon

def setup_env_for_eval(map_name, seed, max_timesteps):
    env = DuckietownEnv(
            map_name=map_name,
            domain_rand=False,
            draw_bbox=False,
            max_steps=max_timesteps,
            seed=seed
    )
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = ImgWrapper(env)  # to rescale images
    env = ModifiedRewardWrapper(env)
    return env

def evaluate_policy(map_name, model, eval_episodes=5, max_timesteps=1500, action_list=None):
    all_seeds = seeds_dict[map_name][:eval_episodes]
    reward_list, length_list = [], []

    for i in range(len(all_seeds)):
        env = setup_env_for_eval(map_name=map_name, seed=all_seeds[i], max_timesteps=max_timesteps)
        state_dim = env.observation_space.shape
        state = env.reset()
        done = False
        steps = 0
        rewards = 0
        past_states = deque(maxlen=5)
        for _ in range(5):
            past_states.append(state)
        new_state = np.stack(past_states)
        new_state = new_state.reshape([-1, state_dim[1], state_dim[2]])

        while not done and steps < max_timesteps:

            if steps == 0:
                prev_pos = env.cur_pos
            else:
                prev_pos = add[0:3]

            # add current position and angle
            add = np.concatenate([env.cur_pos, np.array([env.cur_angle])])

            # add current speed
            add = np.concatenate([add, np.array([env.speed])])

            # add previous position
            add = np.concatenate([add, prev_pos])

            # add distance
            dist = get_dist_to_stop(env, env.cur_pos)
            add = np.concatenate([add, np.array([dist])])

            action_idx = model.act(
                state=new_state,
                epsilon=0,
                additional_data=add,
            )
            action = action_list[action_idx]
            next_state, reward, done, info = env.step(action)
            past_states.appendleft(next_state)

            modified_state = np.stack(past_states)
            modified_state = modified_state.reshape([-1, state_dim[1], state_dim[2]])
            new_state = modified_state

            rewards += reward
            steps += 1

        reward_list.append(rewards)
        length_list.append(steps)

    return reward_list, length_list