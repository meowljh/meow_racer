# import wandb
# # wandb.login()
# # API_KEY = '29cbe68bb0e38908708a92736d3441981d9a1a60'
# """한번 wandb login을 한 이후에는 더 이상 login을 할 필요가 없음."""
# wandb.init(
#             project="racedemia_diffusion_online_rl",
#             name="init-test",
#             dir="log",
#             group="test"
#         )


import gymnasium as gym
import numpy as np

# env = gym.make("MountainCarContinuous-v0", render_mode="human", goal_velocity=0.1)
env = gym.make('Ant-v5', ctrl_cost_weight=0.5, render_mode='human')
obs, _ = env.reset(seed=100, options={"x_init": np.pi, "yinit": 1.})

from tqdm import tqdm

loop = tqdm(range(100))
for i in loop:
    action = env.action_space.sample()
    next_obs, *_ = env.step(action)
env.close() #mujoco 다루려면 이건 필수적으로 있어야 함#