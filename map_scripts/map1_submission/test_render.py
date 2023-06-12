import argparse
import numpy as np
import sys
sys.path.append('gym-duckietown/')
import gym
from gym_duckietown.envs import DuckietownEnv

# # declare the arguments
# parser = argparse.ArgumentParser()

# # Do not change this
# parser.add_argument('--max_steps', type=int, default=2000, help='max_steps')
# # You should set them to different map name and seed accordingly
# parser.add_argument('--map-name', default='map5')
# parser.add_argument('--seed', type=int, default=11, help='random seed')
# args = parser.parse_args()
seeds_dict = {
    "map1": [2, 3, 5, 9, 10],
    "map2": [1, 2, 3, 4, 5, 6, 7, 8],
    "map3": [1, 2, 3, 4, 7, 8, 9, 10],
    "map4": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "map5": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}

def launch_env(MAP_NAME, SEED, MAX_STEPS):
    env = DuckietownEnv(
        map_name=MAP_NAME,
        domain_rand=False,
        draw_bbox=False,
        max_steps=MAX_STEPS,
        seed=SEED,
    )
    print("Initialized environment")
    return env
MAP_NAME = 'map1'
SEED = seeds_dict[MAP_NAME][0]
MAX_STEPS = 2000
env = launch_env(MAP_NAME, SEED, MAX_STEPS)
obs = env.reset()
env.render()
total_reward = 0

# please remove this line for your own policy
actions = np.loadtxt('map1_seed2.txt', delimiter=',')

for (speed, steering) in actions:
    print(steering)
    if steering == 0:
        steering = 0
    obs, reward, done, info = env.step([speed, steering])

    total_reward += reward
    
    print('Steps = %s, Timestep Reward=%.3f, Total Reward=%.3f' % (env.step_count, reward, total_reward))

    env.render()

 
print("Total Reward", total_reward)

# dump the controls using numpy
# np.savetxt('./map5_seed11.txt', actions, delimiter=',')