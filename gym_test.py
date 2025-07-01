import gymnasium as gym
import numpy as np
import os, sys
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
sys.path.append(f"{root}/step_1_rl")
from rl_src.stable.td3 import TD3
from rl_src.stable.common.noise import NormalActionNoise
# from stable_baselines3 import TD3
# from stable_baselines3.common.noise import NormalActionNoise

if __name__ == "__main__":
    env = gym.make("CarRacing-v3", render_mode="human", lap_complete_percent=0.95,
                   domain_randomize=True,
                   continuous=True)
    env.reset()
    
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean = np.zeros(n_actions),
        sigma = 0.001 * np.ones(n_actions)
    )
    model = TD3(
        policy="CnnPolicy",
        env=env,
        action_noise=action_noise,
        verbose=1
    )
    model.learn(total_timesteps=10000, log_interval=10, progress_bar=True)