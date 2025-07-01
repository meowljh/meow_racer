import gymnasium as gym
import os, sys
import numpy as np
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)

from rl_src.stable.common.vec_env import DummyVecEnv
from rl_src.stable.ppo.ppo import PPO
from rl_src.stable.td3.td3 import TD3
from rl_src.stable.ddpg.ddpg import DDPG
from rl_src.stable.sac.sac import SAC
from rl_src.stable.common.noise import NormalActionNoise

def prepare_env(n_envs):
    if n_envs == 1:
        env = gym.make("CarRacing-v3", render_mode="human")
        
    elif n_envs > 1:
        envs = []

        for i in range(n_envs):
            env = gym.make("CarRacing-v3", render_mode="human")
            env.reset(seed = i + 42)
            envs.append(env)
        env = DummyVecEnv(envs)

    observation, info = env.reset(seed=42)
    

    return env

def prepare_model(model_name, env):
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(
            mean = np.zeros(n_actions),
            sigma = 0.1 * np.ones(n_actions)
        ) #action noise를 충분히 더해줘야 초반에 exploration을 충분히 할 수 있음.#
    
    if model_name.upper() == "TD3":

        model = TD3(
            policy="CnnPolicy", env=env, action_noise=action_noise, verbose=1
        )
    
    elif model_name.upper() == "SAC":
        model = SAC(
            policy="CnnPolicy", env=env, action_noise=action_noise, verbose=1
        )
    
    elif model_name.upper() == "PPO":
        model = PPO(
            policy="CnnPolicy", env=env, action_noise=action_noise
        )

    elif model_name.upper() == "DDPG":
        model = DDPG(
            policy="MlpPolicy", env=env, action_noise=action_noise
        )
    return model

def train(model_name, n_envs:int):
    env = prepare_env(n_envs)
    model = prepare_model(model_name, env)

    MAX_EPISODES = 3000
    num_episodes = 0

    ## 최대 episode들 만큼 무조건 주행을 하며 데이터를 모아야 함. ##
    while num_episodes < MAX_EPISODES: 
        #total_timesteps: 하나의 episode에 할당하는 max timestep이라고 생각할 수 있음#
        model.learn(total_timesteps = 10000, log_interval=10, progress_bar=True)
        num_episodes += 1
        
    env.close()

if __name__ == "__main__":
    MODEL_NAME = "SAC"
    ENV_NUM = 2
    train(model_name=MODEL_NAME,
          n_envs=ENV_NUM)


# if __name__ == "__main__":
#     env = gym.make("CarRacing-v3", render_mode="human")
#     n_episodes = 100
#     observation, info = env.reset(seed=42)
    
#     for n in range(n_episodes):
#         env.render()
        
#         action = env.action_space.sample()
#         state, reward, terminated, truncated, info = env.step(action)
        
#         if terminated or truncated:
#             observation, info = env.reset()
    
#     env.close()
    
    
    