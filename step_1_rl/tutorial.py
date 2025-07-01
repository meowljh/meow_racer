import os, sys
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
sys.path.append(f"{root}/rl_src")
from stable.td3 import TD3
from stable.common.noise import NormalActionNoise
# from stable_baselines3.common.noise import NormalActionNoise
from stable.common.vec_env import DummyVecEnv
from stable.common.utils import set_random_seed
 
sys.path.append(os.path.dirname(root))
from step_1_rl.envs.gym_car_racing import Toy_CarRacing
from step_1_rl.envs.gym_car_racing_nam import Toy_CarRacing_NAM
from step_1_rl.envs.gym_car_constants import *


class INP_ARGS:
    def __init__(self):
        self.lidar_deg = 10 #  15
        self.lidar_length = TRACK_WIDTH * 5 #  max(NAM_WINDOW_W, NAM_WINDOW_H)
                    
        self.num_vecs = 5
        self.theta_diff = 20.
        self.do_view_angle = False # True
        
        self.neg_reward_limit = 200
        
        self.progress_time_interval = 30 #timestep#
        self.progress_min_diff = 0.5 #meter#
        
        
def vectorize_env(args, obs_config:dict, rank:int, seed:int=0):
    set_random_seed(seed)
    
    env = Toy_CarRacing(
        observation_config=obs_config,
        do_zoom=3.,
        args=args,
    )    
    env.reset(seed = seed + rank)
    
    return env

if __name__ == "__main__":
    # import gym
    import gymnasium as gym
    num_cpu = 0 # 4 #  2  # 0 # number of processes to use #
    # my_env_id = 'dlwlglP/toy_car_race-v0' 
    # vec_env = SubprocVecEnv(
    #     [vectorize_env(rank=i) for i in range(num_cpu)]
    # )
    args = INP_ARGS()
    
    OBSERV_CONFIG = {
        'dynamic': ['e_c', 'e_phi', 'v_x', 'v_y'],
        # 'dynamic': ['theta', 'e_c', 'e_phi', 'v_x', 'v_y', 'yaw_omega'], ##(1,1,1,1,1,1)## -> yaw_omega: 조향각의 각속도##
        'lidar': [f"lidar_{i}" for i in range(int(180 / args.lidar_deg))], ##(180 / lidar_deg)##
        # 'car': ['omega', 'delta', 'forward'] ##(4, 1, num_vecs)## -> omega: 바퀴 4개의 각속도
        # 'car': ['omega', 'forward'], #delta 값은 결국 조향각이 될텐데, 현재 차량의 조향각보다는 도로에 대한 상대적인 값인 e_phi로 state representation이 가능할 듯 하다.
        # 'car': ['omega', 'delta'], ##우선은 forward observation vector은 제외 해보자.##
        'car': [f"omega_{i}" for i in range(4)] + [f"forward_{i}" for i in range(2*args.num_vecs)]
    } 
    if num_cpu == 1:
        vec_env = DummyVecEnv[
            vectorize_env(args=args, obs_config=OBSERV_CONFIG, rank=0,)]##우선은 하나만, vectorizing 없이 ##
        
    elif num_cpu > 1:
        vec_env = DummyVecEnv(
            [vectorize_env(args=args, obs_config=OBSERV_CONFIG, rank=i,) \
                for i in range(num_cpu)]
        )
    
    elif num_cpu == 0:
        vec_env = Toy_CarRacing(observation_config=OBSERV_CONFIG,
                                do_zoom=3.,
                                args=args)
     
        vec_env.reset()
        
        import pickle
        ### for debugging the limit polygon generation ###
        pickle.dump(vec_env.track_dict, 
                    open('debug_limit.pkl', 'wb'))
        
    n_actions = vec_env.action_space.shape[-1]
    
    action_noise = NormalActionNoise(
        mean = np.zeros(n_actions),
        sigma = 0.1 * np.ones(n_actions)
    ) #action noise를 충분히 더해줘야 초반에 exploration을 충분히 할 수 있음.#
    
    # print(f"Action Noise Check: {isinstance(action_noise, rl_src.stable.common.noise.ActionNoise)}")
    model = TD3(
        policy="MlpPolicy", 
        env=vec_env,  # the environment to learn from #
        action_noise=action_noise, # action noise type helping for hard exploration problem #
        verbose=1
    )
    num_episodes = 0
    MAX_EPISODES = 3000
    
    ## 최대 episode들 만큼 무조건 주행을 하며 데이터를 모아야 함. ##
    while num_episodes < MAX_EPISODES: 
        #total_timesteps: 하나의 episode에 할당하는 max timestep이라고 생각할 수 있음#
        model.learn(total_timesteps = 10000, log_interval=10, progress_bar=True)
        num_episodes += 1
    
    '''[250102] Error fix
    https://github.com/DLR-RM/stable-baselines3/issues/1694'''
    vec_env_for_predict = model.get_env()
    # obs = vec_env.reset()
    obs = vec_env_for_predict.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        # obs, rewards, dones, info = vec_env.step(action)
        # vec_env.render()
        obs, rewards, dones, info = vec_env_for_predict.step(action)
        vec_env_for_predict.render()
        
