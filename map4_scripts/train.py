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
import traceback
import sys

from gym_duckietown.envs import DuckietownEnv
from gym import spaces
from map4_scripts.dqn_model import DeepQNetwork, save_model, load_model, optimize
from itertools import product
from collections import deque
# from wrappers import ResizeWrapper, ImgWrapper, NormalizeWrapper, ModifiedRewardWrapper, NormalizeWrapper
from map4_scripts.wrappers3 import ResizeWrapper, ImgWrapper, ModifiedRewardWrapper, NormalizeWrapper
from map4_scripts.utils import launch_env, seed, ReplayBuffer, compute_epsilon, evaluate_policy
from map4_scripts.feature_engineering import get_dist_to_stop
from map4_scripts.config import custom_params, seeds_dict
from sys import platform
from map4_scripts.Logger import Logger

if platform == 'win32':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def discretize_action_space():
    discrete_v = [0.1, 0.35, 1.0]
    
    denominators = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 4, 8, 12, 16, 20]
    discrete_w = [-np.pi/x for x in denominators] + [-1.0, 0.0, 1.0] + [np.pi/x for x in denominators]
    for x in np.arange(0.01, 0.11, 0.01):
        discrete_w.append(np.round(x, 2))
        discrete_w.append(-np.round(x, 2))

    for x in np.arange(0.001, 0.01, 0.002):
        discrete_w.append(np.round(x, 3))
        discrete_w.append(-np.round(x, 3))
    
    for w in range(len(discrete_w)):
        if discrete_w[w] == 0:
            discrete_w[w] = 0

    action_list = list(product(discrete_v, discrete_w))
    action_list.append((0.0, 1.0))
    action_list.append((0.0, -np.pi/2))
    action_list.append((0.0, np.pi/2))

    action_dim = len(action_list)
    action_dict = {
        tuple(k): i for i, k in enumerate(action_list)
    }  # to return index given an action
    print("Action space:", action_dim)
    print(action_list)

    return action_list, action_dim, action_dict


def train(model_class, MAP_NAME, seeds_dict=seeds_dict):
    torch.cuda.empty_cache()
    logger = Logger(MAP_NAME)
    MODEL_PATH = logger.output_dir
    logger.save_config(custom_params)
    print(MODEL_PATH)
    SEED = seeds_dict[MAP_NAME][0]  # just try the first seed
    env = launch_env(MAP_NAME, SEED)

    # Parameters
    device = custom_params['device']
    restore = custom_params['restore']
    to_save = custom_params['to_save']
    learning_rate = custom_params['learning_rate']
    max_episodes = custom_params['max_episodes']
    t_max = custom_params['t_max']
    min_buffer = custom_params['min_buffer']
    train_steps = custom_params['train_steps']
    batch_size = custom_params['batch_size']
    gamma = custom_params['gamma']
    target_update = custom_params['target_update']
    save_interval = custom_params['save_interval']
    print_interval = custom_params['print_interval']
    eval_interval = custom_params['eval_interval']

    # Wrappers
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = ImgWrapper(env)  # to rescale images
    env = ModifiedRewardWrapper(env)
    print("Initialized wrappers")

    # Set seeds
    seed(SEED)
    state_dim = env.observation_space.shape
    new_state_dim = (state_dim[0]*5, state_dim[1], state_dim[2])
    print("State size:", state_dim)
    print("New state size:", new_state_dim)
    # action_dim = env.action_space.shape[0]

    '''
    Dicretize actions
    '''
    # discretize v
    max_action = float(env.action_space.high[0])
    min_action = float(env.action_space.low[0])
    action_list, action_dim, action_dict = discretize_action_space()
    num_add = 9

    # Initialize model and target network
    if restore:
        model = load_model(os.path.join("./dqn_results", MAP_NAME, "run_{}.pt".format(16000)))
        # model = load_model(os.path.join(MODEL_PATH, "run_{}.pt".format(16000)))
        print("Model restored.")
    else:
        model = model_class(new_state_dim, action_dim, num_additional_inputs=num_add).to(device)

    print(model)
    target = model_class(new_state_dim, action_dim, num_additional_inputs=num_add).to(device)
    target.load_state_dict(model.state_dict())
    target.eval()

    # Initialize replay buffer
    memory = ReplayBuffer()

    # Initialize rewards, losses, and optimizer
    rewards = []
    losses = []
    timesteps = 0
    besttrain_reward = -math.inf
    besteval_reward = -math.inf
    eval_rewards = []
    eval_lengths = []
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for episode in range(max_episodes):
    # for episode in range(2):
        epsilon = compute_epsilon(episode)
        # print(episode, epsilon)
        state = env.reset()
        episode_rewards = 0

        past_states = deque(maxlen=5)
        for _ in range(5):
            past_states.append(state)
        new_state = np.stack(past_states)
        new_state = new_state.reshape([-1, state_dim[1], state_dim[2]])

        for t in range(t_max):
            if t == 0:
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

            # add near-to-stop-sign boolean variable
            # if dist <= 0.31:
            #     add = np.concatenate([add, np.array([0.0])])
            # else:
            #     add = np.concatenate([add, np.array([1.0])])

            # print(add)
            # print(prev_pos, env.cur_pos)

            action_idx = model.act(
                state=new_state,
                epsilon=epsilon,
                additional_data=add,
            )
            action = action_list[action_idx]

            # Apply the action to the environment
            next_state, reward, done, info = env.step(
                action
            )  # action has to be [speed, steering]
            past_states.appendleft(next_state)
            # env.render()
            
            # if episode == 0 and t == 0:
                # import matplotlib.pyplot as plt
                # the resized obs
                # plt.imshow(next_state[0:3,:,:].transpose([1,2,0]))
                # plt.title('Observation resized')
                # plt.show()
                # # road lines obs
                # plt.imshow(next_state[3:,:,:].transpose([1,2,0]))
                # plt.title('Observation road lines')
                # plt.show()
            
            # update state
            modified_state = np.stack(past_states)            
            modified_state = modified_state.reshape([-1, state_dim[1], state_dim[2]])
            memory.add(new_state, modified_state, action, action_idx, reward, done, add)
            
            new_state = modified_state

            if done:
                break

            episode_rewards += reward
            timesteps += 1

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

        if episode % save_interval == 0 and episode != 0 and to_save and len(memory) > min_buffer:
            save_model(model, os.path.join(MODEL_PATH, "run_{}.pt".format(episode)))
            print("Saved", os.path.join(MODEL_PATH, "run_{}.pt".format(episode)))

        # if episode % print_interval == 0:
        if episode % print_interval == 0 and episode > 0:
            print(f"[Episode {episode}]\t TimeSteps {timesteps}\tAvg rewards : {np.mean(rewards[print_interval:]):.3f}"
                    f" Aavg loss: : {np.mean(losses[print_interval * 10 :]):.6f},\t"
                    f" Buffer size : {len(memory)},\tepsilon : {epsilon * 100:.3f}%")
            logger.log_tabular("Episode", episode)
            logger.log_tabular("Timesteps", timesteps)
            logger.log_tabular("AvgReward", np.mean(rewards[print_interval:]))
            logger.log_tabular("AvgLoss", np.mean(losses[print_interval * 10 :]))
            logger.log_tabular("BufferSize", len(memory))
            logger.log_tabular("Epsilon", epsilon*100)
            logger.dump_tabular()

            if np.mean(rewards[print_interval:]) > besttrain_reward:
                besttrain_reward = np.mean(rewards[print_interval:])
                print(f'New best reward {besttrain_reward:.3f}. Storing best_train.pt')
                save_model(model, os.path.join(MODEL_PATH, "best_train.pt"))

        if episode % eval_interval == 0 and episode > 0 and episode > min_buffer:
            eval_reward_list, eval_length_list = evaluate_policy(MAP_NAME,
                            model, eval_episodes=5, max_timesteps=t_max, action_list=action_list)
            eval_rewards.extend(eval_reward_list)
            eval_lengths.extend(eval_length_list)
            mean_eval_reward = np.mean(eval_rewards)
            mean_eval_length = np.mean(eval_lengths)
            print(f'Specific Reward {eval_reward_list}, Length {eval_length_list}')
            print(f" Evaluation Episode {episode} Timesteps {timesteps} AvgReward {mean_eval_reward:.3f}"
                  f" AvgLength {mean_eval_length:.3f}")

            if mean_eval_reward > besteval_reward:
                besteval_reward = mean_eval_reward
                print(f'New best eval reward: {besteval_reward:.3f}. Storing best_eval.pt')
                save_model(model, os.path.join(MODEL_PATH, "best_eval.pt"))

        if episode % eval_interval == 0 or episode % print_interval == 0 and episode > 0:
            print('--------------------------------------')

    return model

if __name__ == "__main__":
    if custom_params['device'] == torch.device('cpu'):
        model = train(DeepQNetwork, MAP_NAME="map4")
    else:
        print(custom_params['device'])
        try:
            from pyvirtualdisplay import Display
            display = Display(visible=0, size=(640, 480))
            display.start()
            model = train(DeepQNetwork, MAP_NAME="map4")
        except Exception as e: # work on python 3.x
            traceback.print_exc()
            print('Interrupted', e)
            display.stop()