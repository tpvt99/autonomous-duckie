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
sys.path.append("../../../gym-duckietown/")

from gym_duckietown.envs import DuckietownEnv
from gym import spaces
from dqn_model import DeepQNetwork, save_model, load_model, optimize
from itertools import product
from collections import deque
from wrappers import ResizeWrapper, ImgWrapper, NormalizeWrapper, ModifiedRewardWrapper, NormalizeWrapper
from utils import launch_env, seed, ReplayBuffer, compute_epsilon
from config import *

MODEL_PATH = "./dqn_results"



def train(model_class, MAP_NAME, MODEL_PATH=MODEL_PATH, seeds_dict=seeds_dict, restore=restore, to_save=to_save):
    torch.cuda.empty_cache()
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    if not os.path.exists(os.path.join(MODEL_PATH, MAP_NAME)):
        os.makedirs(os.path.join(MODEL_PATH, MAP_NAME))
    # MODEL_PATH = os.path.join(MODEL_PATH, '{}_retrain1'.format(MAP_NAME))
    MODEL_PATH = os.path.join(MODEL_PATH, MAP_NAME)
    print(MODEL_PATH)
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    SEED = seeds_dict[MAP_NAME][0]  # just try the first seed

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
    discrete_v = [0.149, 0.35, 0.8, 1.0]
    
    # discretize w
    discrete_w = [-1, -np.pi/4, -np.pi/8, -np.pi/12, -np.pi/16, 0, np.pi/16, np.pi/12, np.pi/8, np.pi/4, 1]

    # discrete_v = [0.149, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # discretize w
    # discrete_w = [-np.pi/2, -np.pi/4, -np.pi/8, -np.pi/16, 0, np.pi/16, np.pi/8, np.pi/4, np.pi/2]
    # step_size = 0.1
    # discrete_w = np.arange(
    #     -2, 2 + step_size / 2, step_size
    # )
    # discrete_v = np.round(discrete_v, 3)
    # discrete_w = np.round(discrete_w, 3)

    for w in range(len(discrete_w)):
        if discrete_w[w] == 0:
            discrete_w[w] = 0
    action_list = list(product(discrete_v, discrete_w))
    action_dim = len(action_list)
    action_dict = {
        tuple(k): i for i, k in enumerate(action_list)
    }  # to return index given an action
    print("Action space:", action_dim)
    print(action_list)

    # Initialize model and target network
    if restore:
        model = load_model(os.path.join("./dqn_results", MAP_NAME, "run_{}.pt".format(2000)))
        print("Model restored.")
    else:
        model = model_class(new_state_dim, action_dim, num_additional_inputs=4).to(device)

    print(model)
    target = model_class(new_state_dim, action_dim, num_additional_inputs=4).to(device)
    target.load_state_dict(model.state_dict())
    target.eval()

    # Initialize replay buffer
    memory = ReplayBuffer()

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

        past_states = deque(maxlen=5)
        for _ in range(5):
            past_states.append(state)
        new_state = np.stack(past_states)
        new_state = new_state.reshape([-1, state_dim[1], state_dim[2]])

        for t in range(t_max):
            add = np.concatenate([env.cur_pos, np.array([env.cur_angle])])
            
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
            print(
                "[Episode {}]\tavg rewards : {:.3f},\tavg loss: : {:.6f},\tbuffer size : {},\tepsilon : {:.1f}%".format(
                    episode,
                    np.mean(rewards[print_interval:]),
                    np.mean(losses[print_interval * 10 :]),
                    len(memory),
                    epsilon * 100,
                )
            )
            with open('out.txt', 'a') as f:
                print(
                "[Episode {}]\tavg rewards : {:.3f},\tavg loss: : {:.6f},\tbuffer size : {},\tepsilon : {:.1f}%".format(
                    episode,
                    np.mean(rewards[print_interval:]),
                    np.mean(losses[print_interval * 10 :]),
                    len(memory),
                    epsilon * 100,
                )
                , file=f)
    return model

if __name__ == "__main__":
    if device == torch.device('cpu'):
        model = train(DeepQNetwork, MAP_NAME="map4")
    else:
        print(device)
        try:
            from pyvirtualdisplay import Display
            display = Display(visible=0, size=(640, 480))
            display.start()
            model = train(DeepQNetwork, MAP_NAME="map4")
        except Exception as e: # work on python 3.x
            traceback.print_exc()
            print('Interrupted', e)
            display.stop()