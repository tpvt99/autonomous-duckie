import cv2
import gym
import numpy as np
from gym import spaces
from config import SCALING_FACTOR
from feature_engineering import select_rgb_white_yellow

class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ResizeWrapper, self).__init__(env)
        # self.shape1 = (
        #     int(env.observation_space.shape[0] * SCALING_FACTOR),
        #     int(env.observation_space.shape[1] * SCALING_FACTOR),
        #     env.observation_space.shape[2] * 2,
        # )
        self.original_shape = env.observation_space.shape
        self.shape = (
            int(env.observation_space.shape[0] * SCALING_FACTOR),
            int(env.observation_space.shape[1] * SCALING_FACTOR),
            # env.observation_space.shape[2] * 2,
            env.observation_space.shape[2],
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

        # crop = observation[100:, :, :]
        # crop1 = select_rgb_white_yellow(observation)
        resized = cv2.resize(
            observation,
            (self.shape[1], self.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        # resized1 = cv2.resize(
        #     crop1,
        #     (self.shape[1], self.shape[0]),
        #     interpolation=cv2.INTER_AREA,
        # )
        # final = np.concatenate((resized, resized1), axis=-1)
        # return final
        # return resized1
        return resized


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
        # except NotInLane:
        except:
            reward = 40.0 * col_penalty
        else:
            # Compute the reward
            reward = (
                + 1.0 * self.speed * lp.dot_dir
                - 10.0 * np.abs(lp.dist)
                + 40.0 * col_penalty
            )

        # penalize 0 and negative speed
        # if self.speed <= 0:
        #     reward -= 10

        """
        Uncomment the below to obey traffic rules
        """
        dist_to_stop = 1000.0

        for obj in self.objects:
            if obj.kind == "sign_stop":
                dist_to_stop = min(dist_to_stop, ((pos[0] - obj.pos[0]) ** 2 + (pos[2] - obj.pos[2]) ** 2) ** 0.5)

        if self.speed > 0.15 and dist_to_stop < 0.3:
            # reward = -100.0
            reward -= 100

        return reward
