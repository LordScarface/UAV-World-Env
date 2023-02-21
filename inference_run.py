import gym
from gym import spaces
from gym.envs.registration import register
import pygame
import numpy as np
from UAV_world import UAV_world
import numpy as np

from stable_baselines3 import PPO

register(
    id='UAV_world/UAV-v0',
    entry_point=UAV_world,
    max_episode_steps=300,
)

n_ac = 5

steps = 2250000

env = gym.make('UAV_world/UAV-v0', num_actors=5, message='After {:,} training Steps'.format(steps).replace(',', '.'))
env.set_render_mode('human')

model = PPO.load("uavs_{}".format(steps))

obs = env.reset()

steps = 0
accum_reward = 0

dones = False

while not dones:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    accum_reward += rewards
    print("Step: {} - Reward: {}, Rewards (accum): {}, Done: {}".format(steps, rewards, accum_reward, dones))
    steps += 1
