import ast
import argparse
import logging
import os
import numpy as np
import random
import math
import gym
import cv2
import torch
import torch.optim as optim
import gym
import sys

# sys.path.append("/Users/lschi/Desktop/admin/MSC_CS/sem2/CS5478/CS4278-5478-Project-Materials/gym-duckietown/")
sys.path.append("gym-duckietown/")

from gym_duckietown.envs import DuckietownEnv
from gym import spaces
from dqn_model2 import DeepQNetwork, save_model, load_model, optimize
from itertools import product
from feature_engineering import select_rgb_white_yellow

"""
Params list
"""
seeds_dict = {
    "map1": [2, 3, 5, 9, 10],
    "map2": [1, 2, 3, 4, 5, 6, 7, 8],
    "map3": [1, 2, 3, 4, 7, 8, 9, 10],
    "map4": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "map5": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}

# to tune this
MAX_STEPS = 2000  # set as a large number for training
t_max = 1500
min_buffer = 99000
epsilon_decay = 32000
buffer_limit = 150000
target_update = 100  # episode(s)
bias = 10

# no need to edit
SCALING_FACTOR = 0.125
learning_rate = 0.0001
gamma = 0.98
batch_size = 32
max_episodes = 200000
print_interval = 20
save_interval = 1000
train_steps = 10
max_epsilon = 1.0
min_epsilon = 0.001

assert buffer_limit >= min_buffer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./dqn_results"

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
    print("Initialized environment")
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
            self.storage.append((state, next_state, action, action_idx, reward, done, add))
        else:
            # Remove random element in the memory beforea adding a new one
            self.storage.pop(random.randrange(len(self.storage)))
            self.storage.append((state, next_state, action, action_idx, reward, done, add))

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

        # state_sample, action_sample, next_state_sample, reward_sample, done_sample
        return {
            "states": torch.from_numpy(np.stack(states)).float().to(device),
            "next_states": torch.from_numpy(np.stack(next_states)).float().to(device),
            "actions": torch.from_numpy(np.stack(actions)).float().to(device),
            "actions_idx": torch.from_numpy(np.stack(actions_idx)).to(device),
            "rewards": torch.from_numpy(np.stack(rewards).reshape(-1, 1)).float().to(device),
            "dones": torch.tensor(dones),
            "adds": torch.from_numpy(np.stack(adds)).to(device)
        }

    def __len__(self):
        return len(self.storage)


def evaluate_policy(env, policy, eval_episodes=10, max_timesteps=500):
    avg_reward = 0.0
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        step = 0
        while not done and step < max_timesteps:
            action = policy.predict(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
            step += 1

    avg_reward /= eval_episodes

    return avg_reward


def compute_epsilon(episode, bias=bias):
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * math.exp(
        -1.0 * (episode + bias) / epsilon_decay
    )
    return epsilon

"""
GYM wrappers
"""
class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ResizeWrapper, self).__init__(env)

        self.original_shape = env.observation_space.shape
        self.shape = (
            int((env.observation_space.shape[0]-100) * SCALING_FACTOR),
            int(env.observation_space.shape[1] * SCALING_FACTOR),
            env.observation_space.shape[2] * 2,
            # env.observation_space.shape[2],
        )
        self.observation_space.shape = self.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            self.shape,
            dtype=self.observation_space.dtype,
        )
        self._proximity_penalty2 = getattr(env, "_proximity_penalty2", None)
        self.get_lane_pos2 = getattr(env, "get_lane_pos2", None)

    def observation(self, observation):

        crop = observation[100:, :, :]
        crop1 = select_rgb_white_yellow(crop)
        resized = cv2.resize(
            crop,
            (self.shape[1], self.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        resized1 = cv2.resize(
            crop1,
            (self.shape[1], self.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        final = np.concatenate((resized, resized1), axis=-1)
        return final


class NormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizeWrapper, self).__init__(env)
        self.obs_lo = self.observation_space.low[0, 0, 0]
        self.obs_hi = self.observation_space.high[0, 0, 0]
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(0.0, 1.0, obs_shape, dtype=np.float32)
        self._proximity_penalty2 = getattr(env, "_proximity_penalty2", None)
        self.get_lane_pos2 = getattr(env, "get_lane_pos2", None)

    def observation(self, obs):
        if self.obs_lo == 0.0 and self.obs_hi == 1.0:
            return obs
        else:
            return (obs - self.obs_lo) / (self.obs_hi - self.obs_lo)


class ImgWrapper(gym.ObservationWrapper):
    # to have [C, H, W]
    def __init__(self, env=None):
        super(ImgWrapper, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype,
        )
        self._proximity_penalty2 = getattr(env, "_proximity_penalty2", None)
        self.get_lane_pos2 = getattr(env, "get_lane_pos2", None)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class NotInLane(Exception):
    """ Raised when the Duckiebot is not in a lane. """

    pass


class ModifiedRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(ModifiedRewardWrapper, self).__init__(env)
        # self.env = env
        self._proximity_penalty2 = getattr(env, "_proximity_penalty2", None)
        self.get_lane_pos2 = getattr(env, "get_lane_pos2", None)

    def reward(self, reward):
        """
        Lane following only
        """
        # pos = self.env.cur_pos
        # angle = self.env.cur_angle
        pos = self.cur_pos
        angle = self.cur_angle
        # Compute the collision avoidance penalty
        # col_penalty = self.env._proximity_penalty2(pos, angle)
        col_penalty = self._proximity_penalty2(pos, angle)

        # Get the position relative to the right lane tangent
        try:
            # lp = self.env.get_lane_pos2(pos, angle)
            lp = self.get_lane_pos2(pos, angle)
        except NotInLane:
            reward = 40 * col_penalty
        else:
            # Compute the reward
            reward = (
                +1.0 * self.speed * lp.dot_dir
                + -10 * np.abs(lp.dist)
                + +40 * col_penalty
            )

        # penalize 0 and negative speed
        if self.speed <= 0:
            reward -= 10

        """
        Uncomment the below to obey traffic rules
        """
        # dist_to_stop = 1000.0

        # for obj in self.objects:
        #     if obj.kind == "sign_stop":
        #         dist_to_stop = min(dist_to_stop, ((pos[0] - obj.pos[0]) ** 2 + (pos[2] - obj.pos[2]) ** 2) ** 0.5)

        # if self.speed > 0.15 and dist_to_stop < 0.3:
        #     reward = -100.0

        return reward

### MY DQN HERE ###
def train(model_class, MAP_NAME, MODEL_PATH=MODEL_PATH, seeds_dict=seeds_dict):
    torch.cuda.empty_cache()
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    if not os.path.exists(os.path.join(MODEL_PATH, MAP_NAME)):
        os.makedirs(os.path.join(MODEL_PATH, MAP_NAME))
    MODEL_PATH = os.path.join(MODEL_PATH, MAP_NAME)
    # MODEL_PATH = os.path.join(MODEL_PATH, '{}_more_actions'.format(MAP_NAME))
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    SEED = seeds_dict[MAP_NAME][0]  # just try the first seed

    restore = False
    to_save = True
    env = launch_env(MAP_NAME, SEED)
    # Wrappers
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = ImgWrapper(env)  # to rescale images
    env = ModifiedRewardWrapper(env)
    print("Initialized wrappers")

    # Set seeds
    seed(SEED)
    state_dim = env.observation_space.shape
    print("State size:", state_dim)
    # action_dim = env.action_space.shape[0]

    # Discretize action space. Or use DiscreteWrapper.
    max_action = float(env.action_space.high[0])
    min_action = float(env.action_space.low[0])
    step_size = 0.2
    discrete_v = np.arange(0, max_action + step_size / 2, step_size)  # velocity. [0, 1]
    discrete_v = [k for k in discrete_v if k != 0]
    discrete_w = np.arange(
        min_action, max_action + step_size / 2, step_size
    )  # steer angle [-1, 1]
    discrete_v = np.round(discrete_v, 2)
    discrete_w = np.round(discrete_w, 2)
    for w in range(len(discrete_w)):
        if discrete_w[w] == 0:
            discrete_w[w] = 0
    action_list = list(product(discrete_v, discrete_w))
    action_dim = len(action_list)
    action_dict = {
        tuple(k): i for i, k in enumerate(action_list)
    }  # to return index given an action
    print("Action space:", action_dim)

    # Initialize model and target network
    if restore:
        # model = load_model(os.path.join(MODEL_PATH, "run_{}.pt".format(15000)))
        print("Model restored.")
    else:
        model = model_class(state_dim, action_dim, num_additional_inputs=4).to(device)

    target = model_class(state_dim, action_dim, num_additional_inputs=4).to(device)
    target.load_state_dict(model.state_dict())
    target.eval()

    # Initialize replay buffer
    memory = ReplayBuffer()

    print(model)

    # Initialize rewards, losses, and optimizer
    rewards = []
    losses = []
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for episode in range(max_episodes):
    # for episode in range(2):
        epsilon = compute_epsilon(episode)
        state = env.reset()
        episode_rewards = 0
        env.render()
        for t in range(t_max):
        # for i in range(5):
            add = np.concatenate([env.cur_pos, np.array([env.cur_angle])])
            
            action_idx = model.act(
                state=state,
                epsilon=epsilon,
                additional_data=add,
            )
            action = action_list[action_idx]

            # Apply the action to the environment
            next_state, reward, done, info = env.step(
                action
            )  # action has to be [speed, steering]

            # if episode == 0 and t == 0:
            #   import matplotlib.pyplot as plt
            #   # the resized obs
            #   plt.imshow(next_state[0:3,:,:].transpose([1,2,0]))
            #   plt.title('Observation resized')
            #   plt.show()
            #   # road lines obs
            #   plt.imshow(next_state[3:,:,:].transpose([1,2,0]))
            #   plt.title('Observation road lines')
            #   plt.show()
            
            # Save transition to replay buffer
            memory.add(state, next_state, action, action_idx, reward, done, add)
            state = next_state

            if done:
                break

            episode_rewards += reward

        rewards.append(episode_rewards)

        # Train the model if memory is sufficient
        if len(memory) > min_buffer:
            for i in range(train_steps):
                # for i in range(2):
                loss = optimize(model, target, memory, optimizer, batch_size, gamma)
                losses.append(loss.item())

        # Update target network every once in a while
        if episode % target_update == 0:
            target.load_state_dict(model.state_dict())

        if episode % save_interval == 0 and episode != 0 and to_save:
        # if episode % save_interval == 0:
            # if episode % save_interval == 0:
            save_model(model, os.path.join(MODEL_PATH, "run_{}.pt".format(episode)))
            print("Saved", os.path.join(MODEL_PATH, "run_{}.pt".format(episode)))

        # if episode % print_interval == 0:
        if episode % print_interval == 0 and episode > 0:
            print(
                "[Episode {}]\tavg rewards : {:.3f},\tavg loss: : {:.6f},\tbuffer size : {},\tepsilon : {:.1f}%".format(
                    episode,
                    np.mean(rewards[print_interval:]),
                    np.mean(losses[print_interval * 10 :]),
                    len(memory),
                    epsilon * 100,
                )
            )
    return model

if __name__ == "__main__":
    if device == 'cuda':
        from pyvirtualdisplay import Display
        display = Display(visible=0, size=(640, 480))
        display.start()
    model = train(DeepQNetwork, MAP_NAME="map1")