import os, sys
import numpy as np
from gymnasium.spaces import Box, Dict

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
from rl_src.stable.her.her_replay_buffer import HerReplayBuffer

def load_replay_buffer(name, env):
    if name is None:
        return name
    elif name == 'HerReplayBuffer':
        replay_buffer_kwargs = {
            'env': env, 
            # 'buffer_size': int(1e+6),
            # 'observation_space': Dict(dict_obs),
            # 'action_space': env.action_space
        }
        return replay_buffer_kwargs
        # return HerReplayBuffer(
        #     buffer_size=int(1e+6),
        #     observation_space=Dict(dict_obs),
        #     action_space=env.action_space,
        #     n_envs=1, 
        #     env=env
        # )
        