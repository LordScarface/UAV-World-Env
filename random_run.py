import gym
from gym import spaces
from gym.envs.registration import register
import pygame
import numpy as np
from UAV_world import UAV_world
import numpy as np

register(
    id='UAV_world/UAV-v0',
    entry_point=UAV_world,
    max_episode_steps=300,
)

n_ac = 5

env = gym.make('UAV_world/UAV-v0', num_actors=n_ac)
env.set_render_mode('human')
env.reset()

while True:
    env.step(np.random.randint(0,4,size=n_ac))