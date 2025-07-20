import os, sys

import gymnasium as gym
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gym_env_wrapper import MeowGymWrapper

def build_gym_env(env_name:str, render_mode:str, seed:int):
    env = gym.make(id=env_name, render_mode=render_mode)
    env.reset(seed=seed)
    env = MeowGymWrapper(env=env)
    action_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]

    return env, obs_dim, action_dim



def build_racedemia_env(env_name:str, seed:int):
    return