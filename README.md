# autonomous-duckie

A work-in-progress implementation of Almási, P., Moni, R., & Gyires-Tóth, B. (2020). Robust Reinforcement Learning-based Autonomous Driving Agent for Simulation and Real World. Proceedings of the International Joint Conference on Neural Networks. https://doi.org/10.1109/IJCNN48605.2020.9207497.

The focus of this DQN is for lane-following **only** (for now).

# Guide
Add `.py` files into the `CS4278-5478-Project-Materials/` folder to run `train2.py`.

# To-do list
1) Have not concatenated historial observations to DQN inputs.
2) The paper's model is incredibly small (4MB), not too sure if its too small to learn anything.
3) Have not applied STOP sign traffic rules into the reward shaping.
4) May have to rethink what *t_max* is, may need to check with Prof. It is the maximum number of timesteps to run duckiebot in a single map.
5) Have not given more reward for faster speed. We want the duck to move as fast as possible when there are no obstacles/ STOP sign in view.
