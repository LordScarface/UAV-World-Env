import pygame
import numpy as np

import gym
from gym import spaces
from gym.envs.registration import register

from UAV_world import UAV_world

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan



register(
    id='UAV_world/UAV-v0',
    entry_point=UAV_world,
    reward_threshold=1,
    max_episode_steps=1000,
)

first_run = True

env = make_vec_env('UAV_world/UAV-v0', n_envs=4)

model = PPO("MultiInputPolicy", env, verbose=2, tensorboard_log='./tensorboard', device='auto')

TIMESTEPS = 250000
iters = 0
for i in range(50):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=first_run)
    model.save(f"./uavs_{TIMESTEPS*i}")
    first_run = False


