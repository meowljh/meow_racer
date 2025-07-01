import os, sys
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
import numpy as np
import torch
import pickle
sys.path.append(f"{root}/rl_src")
import copy
## TD3와 SAC 모두 off-policy (data sampling의 효율성 증대)
from stable.td3 import TD3
from stable.sac import SAC ## source code에 있는 SAC 사용!
from stable.common.noise import NormalActionNoise
from loader import load_replay_buffer
# from stable_baselines3.common.noise import NormalActionNoise
from stable.common.vec_env import DummyVecEnv
from stable.common.nstep_buffers import NStepReplayBuffer
from stable.common.lambda_step_buffers import LambdaStepReplayBuffer
from stable.common.utils import set_random_seed
from stable.her.her_replay_buffer import HerReplayBuffer
from natsort import natsorted
from glob import glob

sys.path.append(os.path.dirname(root))

from evaluate import evaluate
from step_1_rl.envs.gym_car_racing import Toy_CarRacing
from step_1_rl.envs.gym_car_racing_nam import Toy_CarRacing_NAM
from step_1_rl.envs.gym_car_constants import *

import argparse

from load_rl_model import load_rl_algo

def int2bool(i):
    return True if i == 1 else False

def train_freq(s):
    try:
        freq_int, freq_type = map(str, s.split(','))
        freq_int = int(freq_int)
        return (freq_int, freq_type)
    except:
        raise argparse.ArgumentTypeError("Argument train_freq should contain 2 values")
    
def nullable_string(val):
    if not val:
        return None
    return val

def set_random_seed(seed):
    '''gym environment는 reset하는 과정에서 random seed 세팅도 함'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    
def load_args():
    parser = argparse.ArgumentParser()
    ######################################################
    ######## for noise scheduler ###########
    parser.add_argument("--with_tile_reward", type=int, default=0)
    parser.add_argument("--am_with_theta_reward", type=int, default=0)
    parser.add_argument("--noise_change_interval", type=str, default="episode_1000")

    parser.add_argument("--final_sigma", type=float, default=0.)
    parser.add_argument("--max_gaussian_step", type=float, default=1e+7)
    
    parser.add_argument("--action_num", type=int, default=3)
    
    parser.add_argument("--use_rotated_forward", type=int, default=0)
    
    parser.add_argument("--is_aip", type=int, required=True)
    
    parser.add_argument("--random_seed", type=int, required=True)
    parser.add_argument("--rl_algorithm", type=str, required=True)
    parser.add_argument("--tau", type=float, default=0.005)
    
    parser.add_argument("--action_noise_type", type=str, default="gaussian")
    parser.add_argument("--action_noise_sigma", type=float, default=0.1)
    ############## add argument for Beta ######################
    parser.add_argument("--use_beta_dist", type=int, default=0)
    ############### add argument for PPO #######################
    
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--clip_range_vf", type=float, default=-1)
    parser.add_argument("--normalize_advantage", type=int, default=1)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    
    ############################################################
    parser.add_argument("--tensorboard_folder", type=str, default='')
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--finetune_path", type=str, default=None)
    parser.add_argument("--do_penalty_max_reward", type=int, default=0)
    parser.add_argument("--do_reverse", type=float, default=0.)
    parser.add_argument("--random_start", type=int, default=0)
    parser.add_argument("--oscillation_max_penalty", type=float, default=0.)
    parser.add_argument("--center_line_far_max_reward_corner", type=float, default=0.)
    
    parser.add_argument("--n_steps_td", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr_scheduler_config", type=nullable_string, default='')
    parser.add_argument("--center_line_max_penalty", type=float, default=0.)
    
    parser.add_argument("--do_reverse_nam", type=float, default=0.)

    parser.add_argument("--oscillation_max_reward_corner", type=float, default=0.)
    
    parser.add_argument("--oscillation_penalty", type=float, default=0., help="The maximum value is 1. and the minimum value is 0.")
    parser.add_argument("--straight_kappa_limit", type=float, default=0.01)
    ##########################################################################
    #################### configuration for track type ########################
    parser.add_argument("--env_type", type=str, default="random")
    parser.add_argument("--skip_rate", type=int, default=1)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--max_reward_tile", type=int, default=5000)
    parser.add_argument("--both_track_ratio", type=float, default=0.0)
    parser.add_argument("--use_jw", type=int, default=0)
    parser.add_argument("--use_theta_diff", type=int, default=0)
    parser.add_argument("--min_theta_movement", type=float, default=0.0)
    parser.add_argument("--nam_width_change", type=int, default=0)
    parser.add_argument("--num_random_checkpoints", type=int, default=-1)
    parser.add_argument("--max_both_track_ratio", type=float, default=-1)
    parser.add_argument("--center_line_far_max_reward", type=float, default=0.)
    #########################################################################
    ######################### configuration for SAC #########################
    parser.add_argument("--use_sde", type=int, default=0)
    parser.add_argument("--total_timesteps", type=int, default=10000)
    parser.add_argument("--learning_start", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--replay_buffer_class", type=str, default=None)
    parser.add_argument("--mpc_reward_scaler", type=float, default=1.)
    parser.add_argument("--replay_buffer_size", type=int, default=1_000_000)
    parser.add_argument("--ent_coef", type=str, default='auto', help="Entropy regularization coefficient for reward scaling")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor for calculating the value")
    parser.add_argument("--train_freq", dest="train_freq", type=train_freq, default=(1, "step"))
    parser.add_argument("--gradient_step", type=int, default=1, help="number of gradient steps to do after each rollout of training (if set to -1, do as many gradient steps as the number of steps)")
    parser.add_argument("--new_friction_limit", type=float, default=1.)
    parser.add_argument("--consider_forward_vec", type=int, default=1,
                        help="whether to consider the forward vector length when checking the oscillation"
    )
    #########################################################################
    #################### configuration for training #########################
    parser.add_argument("--weight_save_converge", type=int, default=0)
    parser.add_argument("--weight_save_interval", type=int, default=0)
    
    parser.add_argument("--n_envs", type=int, default=0)
    parser.add_argument("--screen_title", type=str, default='')
    parser.add_argument("--random_checkpoints", action="store_true", default=False)

    parser.add_argument("--vel_weight", type=float)
    parser.add_argument("--theta_weight", type=float)
    parser.add_argument("--limit_weight", type=float)

    # parser.add_argument("--terminate_out", action="store_true", default=False)
    parser.add_argument("--terminate_out", type=int)

    parser.add_argument("--reward_type", type=str, required=True)
    parser.add_argument("--time_penalty", type=float, required=True)
    # parser.add_argument("--same_tile_penalty", action="store_true", default=False)
    parser.add_argument("--same_tile_penalty", type=int, default=0)
    parser.add_argument("--min_movement", type=float, default=1e-3)
    parser.add_argument("--terminate_penalty", type=float, default=200.)

    parser.add_argument("--min_vel", type=float)
    parser.add_argument("--tile_leave_weight", type=float, default=0.)

    parser.add_argument("--save", type=str, default=None)

    
    parser.add_argument("--step_lambda", type=float, default=0.)
    parser.add_argument("--step_lambda_mode", type=str, default="forward")
    
    parser.add_argument("--backward_penalty", type=float, default=100)
    
    ##########################################################################
    ############## configuration for observation space #######################
    parser.add_argument("--num_vecs", type=int, default=10, help="number of vectors used for the forward vectors")
    parser.add_argument("--theta_diff", type=int, default=10, help="distance between the forward vector target positions")
    parser.add_argument("--lidar_deg", type=int, default=10)
    
    parser.add_argument("--use_curvature", type=int, default=0, help="wether to use the curvature of the center line of the track for the observation space")
    
    
    
    parser.add_argument("--use_steer", action="store_true", default=False)
    parser.add_argument("--use_gas", action="store_true", default=False)
    parser.add_argument("--use_brake", action="store_true", default=False)
    parser.add_argument("--use_force", action="store_true", default=False)
    parser.add_argument("--use_delta", action="store_true", default=False)
    
    
    parser.add_argument("--body_left_penalty", type=float, default=0.)
    parser.add_argument("--body_left_mode", type=str, default="constant")
    
    parser.add_argument("--ec_weight", type=float, default=0.)
    parser.add_argument("--etheta_weight", type=float, default=0.)
    loaded = parser.parse_args()

    loaded.terminate_out = int2bool(loaded.terminate_out)
    loaded.is_aip = int2bool(loaded.is_aip)
    loaded.use_jw = int2bool(loaded.use_jw)
    loaded.use_sde = int2bool(loaded.use_sde)
    loaded.weight_save_converge = int2bool(loaded.weight_save_converge)
    loaded.use_theta_diff = int2bool(loaded.use_theta_diff)
    loaded.debug = int2bool(loaded.debug)
    loaded.nam_width_change = int2bool(loaded.nam_width_change)
    loaded.random_start = int2bool(loaded.random_start)
    
    loaded.do_penalty_max_reward = int2bool(loaded.do_penalty_max_reward)
    loaded.consider_forward_vec = int2bool(loaded.consider_forward_vec)
    
    loaded.use_beta_dist = int2bool(loaded.use_beta_dist)
    
    loaded.use_rotated_forward = int2bool(loaded.use_rotated_forward)
    
    loaded.normalize_advantage = int2bool(loaded.normalize_advantage)
    
    loaded.am_with_theta_reward = int2bool(loaded.am_with_theta_reward)
    
    loaded.with_tile_reward = int2bool(loaded.with_tile_reward)
    
    if loaded.clip_range_vf == -1:
        loaded.clip_range_vf = None
    if loaded.oscillation_penalty > 1.:
        raise UserWarning("Oscillation Penalty weight should not exceed 1..")
    loaded.oscillation_penalty = np.clip(loaded.oscillation_penalty, 0., 1.)
    
    loaded.max_both_track_ratio = loaded.both_track_ratio if loaded.max_both_track_ratio == -1 else loaded.max_both_track_ratio
    
    loaded.use_curvature = int2bool(loaded.use_curvature)
    
    return loaded

class INP_ARGS:
    def __init__(self):
        self.am_with_theta_reward = False
        self.with_tile_reward = False
        
        self.noise_change_interval = "episode_1000"
        self.action_num = 3
        self.use_rotated_forward = False
        
        self.final_sigma = 0.
        self.max_gaussian_step = 1e+7
        self.is_aip = False
        self.use_curvature = False
        
        self.action_noise_type = "gaussian"
        self.action_noise_sigma = 0.1
        self.use_beta_dist = False
        
        self.random_seed = 42
        self.rl_algorithm = 'SAC'
        
        self.tensorboard_folder = ''
        self.random_checkpoints = False
        self.save = ''
        self.backward_penalty = 100.
        self.screen_title = ''
        
        self.lidar_deg = 10 #  15
        self.lidar_length = TRACK_WIDTH * 5 #  max(NAM_WINDOW_W, NAM_WINDOW_H)
                    
        self.num_vecs = 10
        self.theta_diff = 10.
        self.tau = 0.005 # for SAC
        self.do_view_angle = False # True
        
        self.neg_reward_limit = 200
        
        self.progress_time_interval = 30 #timestep#
        self.progress_min_diff = 0.5 #meter#
        
        self.max_reward = 1000. # maximum number of tiles controlled #
        
        self.vel_weight = 1. # 0.1 # 10. # 0.1 # weighting parameter for the velocity of the car #
        self.theta_weight = 5. # 20. # weighting parameter for the total travel distance difference along the center line #
        self.limit_weight = 20. # weighting parameter for leaving the track just a little bit #

        self.terminate_out = True 
        self.terminate_penalty = 200.

        self.do_view_angle = False

        self.time_penalty = 0.1
        self.reward_type = "baseline"
        self.same_tile_penalty = 0
        self.min_movement = 0.5
        self.min_vel = 0.
        self.tile_leave_weight = 0.

        self.n_envs = 0
        self.screen_title = ''
        
        self.skip_rate = 1
        self.max_episodes = 3000
        self.max_reward_tile = 5000
        self.both_track_ratio = 0.0
        
        self.replay_buffer_class = None
        self.replay_buffer_size = int(1e+6)
        # self.ent_coef = 0.1
        self.ent_coef = 'auto_0.1'
        self.gamma = 0.99
        self.train_freq = (1, "step") #매 step마다 학습 시키는게 과연 맞을까?
        self.gradient_step = 1
        self.new_friction_limit= 1.
        self.total_timesteps = 10000
        self.learning_start = 100
        self.learning_rate = 3e-4
        self.use_jw = False
        self.use_sde = False
        
        self.weight_save_converge = False
        self.weight_save_interval = 0
        self.use_theta_diff = False
        self.debug = False
        self.do_reverse = 0.
        self.do_reverse_nam = 0.
        self.min_theta_movement = 0
        self.finetune_path = None
        
        self.nam_width_change = False
        self.num_random_checkpoints = -1
        self.random_start = False
        self.max_both_track_ratio = -1
        
        self.oscillation_penalty = 0.
        self.straight_kappa_limit = 0.01
        
        self.center_line_max_penalty = 0.
        
        self.batch_size = 256
        self.n_steps_td = 0
        self.do_penalty_max_reward = False
        
        self.body_left_penalty = 0.
        self.body_left_mode = 'constant'
        
        self.step_lambda = 0.
        self.step_lambda_mode = 'forward'
        
        self.ec_weight = 0.
        self.etheta_weight = 0.
        
        self.consider_forward_vec  =True
        self.mpc_reward_scaler = 1.
        
        self.oscillation_max_penalty = 0.
        self.oscillation_max_reward_corner = 0.
        self.center_line_far_max_reward = 0.
        self.center_line_far_max_reward_corner = 0.
        
        
        self.n_steps=2048
        self.n_epochs=10
        self.gae_lambda=0.95
        self.clip_range=0.2
        self.clip_range_vf=None
        self.normalize_advantage=True
        self.vf_coef= 0.5
        self.max_grad_norm=0.5
        self.target_kl = None
        
def vectorize_env(env_type:str, args, obs_config:dict, rank:int, seed:int=0):
    set_random_seed(seed)

    if env_type == "random":
        env = Toy_CarRacing(
            observation_config=obs_config,
            lap_complete_percent=1.,
            do_zoom=3.,
            args=args,
        )    
    elif env_type == "nam":
        env = Toy_CarRacing_NAM(
            observation_config=obs_config,
            lap_complete_percent=1.,
            do_zoom=3.,
            args=args
        )
    env.reset(seed = seed + rank)
    
    return env

def prepare_args():
    args = INP_ARGS()
    use_args = load_args()

    for key, value in args.__dict__.items():
        if hasattr(use_args, key) and use_args.__dict__[key] is not None:
            setattr(args, key, use_args.__dict__[key]) 
            # args.__dict__[key] = value

        print(f"{key} ---> {args.__dict__[key]}")
    
    '''DEFINE THE OBSERVATION CONFIGURATIONS
    What observation values could be useful?
    Use the top-down view image as well?
    Change the numerical observation states to an image??
    '''
    OBSERV_CONFIG = {
        'dynamic': ['e_c', 'e_phi', 'v_x', 'v_y'],
        # 'dynamic': ['theta', 'e_c', 'e_phi', 'v_x', 'v_y', 'yaw_omega'], ##(1,1,1,1,1,1)## -> yaw_omega: 조향각의 각속도##
        'lidar': [f"lidar_{i}" for i in range(int(180 / args.lidar_deg))], ##(180 / lidar_deg)##
        # 'car': ['omega', 'delta', 'forward'] ##(4, 1, num_vecs)## -> omega: 바퀴 4개의 각속도
        # 'car': ['omega', 'forward'], #delta 값은 결국 조향각이 될텐데, 현재 차량의 조향각보다는 도로에 대한 상대적인 값인 e_phi로 state representation이 가능할 듯 하다.
        # 'car': ['omega', 'delta'], ##우선은 forward observation vector은 제외 해보자.##
        'car': [f"omega_{i}" for i in range(4)] + \
                [f"forward_{i}" for i in range(2*args.num_vecs)] , ## (x, y)의 2차원 벡터의 형태이기 때문에 2*num_vecs의 개수만큼 forward vector을 저장하게 된다.
         
    } 
    
    if use_args.use_brake:
        OBSERV_CONFIG['car'].append("brake")
    if use_args.use_gas:
        OBSERV_CONFIG['car'].append("gas")
    if use_args.use_steer:
        OBSERV_CONFIG['car'].append("steer")
    if use_args.use_delta:
        OBSERV_CONFIG['car'].append("delta")
    if use_args.use_force:
        # OBSERV_CONFIG['car'].extend([f"force_{i}" for i in range(4)])
        OBSERV_CONFIG['car'].append("force")
        
    if use_args.use_curvature: #curvature입력도 num_vecs의 개수만큼 theta_diff의 크기의 차이를 갖고 하면 될듯
        OBSERV_CONFIG['car'].extend([f"curvature_{i}"  for i in range(args.num_vecs)])

    return args, use_args, OBSERV_CONFIG


if __name__ == "__main__":
    # import gym
    import gymnasium as gym
    # num_cpu = 0 # 8 #  0 # 4 #  2  # 0 # number of processes to use #
    # my_env_id = 'dlwlglP/toy_car_race-v0' 
    # vec_env = SubprocVecEnv(
    #     [vectorize_env(rank=i) for i in range(num_cpu)]
    # )

    args, use_args, OBSERV_CONFIG = prepare_args()
    nam_env = Toy_CarRacing_NAM(
        observation_config=OBSERV_CONFIG,
        lap_complete_percent=1.,
        do_zoom=3.,
        args=args
    )

    num_cpu = use_args.n_envs

    if num_cpu == 1:
        vec_env = DummyVecEnv[
            vectorize_env(env_type=use_args.env_type, args=args, obs_config=OBSERV_CONFIG, rank=0,)
            ]##우선은 하나만, vectorizing 없이 ##
        
    elif num_cpu > 1:
        vec_env = DummyVecEnv(
            [vectorize_env(env_type=use_args.env_type, args=args, obs_config=OBSERV_CONFIG, rank=i,) \
                for i in range(num_cpu)]
        )
    
    elif num_cpu == 0:
        if use_args.env_type == 'random':
            vec_env = Toy_CarRacing(observation_config=OBSERV_CONFIG,
                                lap_complete_percent=1.,
                                do_zoom=3.,
                                # do_zoom=1.,
                                args=args)
        else:
            vec_env = Toy_CarRacing_NAM(
                observation_config=OBSERV_CONFIG,
                lap_complete_percent=1.,
                do_zoom=3.,
                args=args
            )
     
        vec_env.reset()
        
        # import pickle
        # ### for debugging the limit polygon generation ###
        # pickle.dump(vec_env.track_dict, 
        #             open('debug_limit.pkl', 'wb'))
        
    n_actions = vec_env.action_space.shape[-1]
    
    action_noise = NormalActionNoise(
        mean = np.zeros(n_actions),
        sigma = 0.1 * np.ones(n_actions)
    ) #action noise를 충분히 더해줘야 초반에 exploration을 충분히 할 수 있음.#
    
    # replay_buffer_kwargs = None
    
    # if args.n_steps_td > 0:
    #     if args.step_lambda > 0.:
    #         replay_buffer_class = LambdaStepReplayBuffer

    #     else:   
    #         replay_buffer_class = NStepReplayBuffer
         
    # elif args.replay_buffer_class == 'HerReplayBuffer':
    #     replay_buffer_class = None
    #     replay_buffer_kwargs = load_replay_buffer(args.replay_buffer_class, env=vec_env)   
    # else:
    #     replay_buffer_class = None
        
    # # print(f"Action Noise Check: {isinstance(action_noise, rl_src.stable.common.noise.ActionNoise)}")
    # # model = TD3
    # model = SAC(
    #     policy="MultiInputPolicy" if args.replay_buffer_class=='HerReplayBuffer' else "MlpPolicy", 
    #     env=vec_env,  # the environment to learn from #
    #     action_noise=action_noise, # action noise type helping for hard exploration problem #
    #     verbose=1,
    #     buffer_size=args.replay_buffer_size,
    #     batch_size=args.batch_size,
    #     # replay_buffer_class=HerReplayBuffer if args.replay_buffer_class=='HerReplayBuffer' else None,
    #     # replay_buffer_class = NStepReplayBuffer if args.n_steps_td > 0 else None,
    #     # replay_buffer_kwargs=load_replay_buffer(args.replay_buffer_class, env=vec_env) if args.replay_buffer_class=='HerReplayBuffer' else None,
    #     replay_buffer_class=replay_buffer_class,
    #     replay_buffer_kwargs=replay_buffer_kwargs,
    #     ent_coef=args.ent_coef,
    #     gamma=args.gamma,
    #     train_freq=args.train_freq,
    #     gradient_steps=args.gradient_step,
    #     learning_starts=args.learning_start,
    #     learning_rate=args.learning_rate,
    #     use_sde=args.use_sde,
        
    #     n_steps_td=args.n_steps_td,
        
    #     step_lambda=args.step_lambda,
    #     step_lambda_mode=args.step_lambda_mode
    # )
    
    model = load_rl_algo(args, vec_env=vec_env)
    
    # exp_path = f"{os.path.dirname(os.path.abspath(__file__))}/experiments/{args.save}".replace("\\", "/")
    '''change experiment path to D Drive (CPU memory issue)
    and to /nas for AIP usage'''
    if args.is_aip:
        exp_path = f"/nas/{args.save}"
    else:
        exp_path = f"D:/gen_data_experiments/{args.save}"
    os.makedirs(exp_path, exist_ok=True)
    
    '''add additional parameters for the trained model which can be loaded from the *.zip file'''
    train_onload = False
    last_episode = None
    if use_args.finetune_path is not None:
        # breakpoint()
        # fpath = f"{os.path.dirname(os.path.abspath(__file__))}/experiments/{args.finetune_path}/{args.finetune_path}.zip"
        fpath = f"{os.path.dirname(exp_path)}/{args.finetune_path}/{args.finetune_path}.zip"
        print("Finetune Model exists...")
        print("Loading trained model for fine tune...")
        model.set_env(vec_env)
        model = model.load(fpath, env=vec_env,
                           custom_objects = {'learning_rate': use_args.learning_rate})
        train_onload = True
        
    elif os.path.isfile(os.path.dirname(exp_path) + f"/{args.save}" ".zip"):
        print("Model exists...")
        print("Loading trained  model...")
        model.set_env(vec_env)
        model = model.load(f"{os.path.dirname(exp_path)}/{args.save}/{args.save}.zip", # f"experiments/{args.save}/{args.save}.zip", 
                           env=vec_env,
                           custom_objects = {'learning_rate': use_args.learning_rate})
                           
        train_onload = True 
        saved_episodes = natsorted(glob(exp_path + f'/*_log_weight.zip'))
 
        last_episode = saved_episodes[-1].replace('\\', '/').split('/')[-1].split('_')[0]
        last_episode = int(last_episode)
        print("==========================================")
        print(f"Starting episode again from #{last_episode}")
        print("==========================================")
        
    # model.learning_rate = use_args.learning_rate
        
    num_episodes = 0 if last_episode is None else last_episode
    MAX_EPISODES = 3000 if args.max_episodes is None else args.max_episodes
    
    ## 최대 episode들 만큼 무조건 주행을 하며 데이터를 모아야 함. ##

    best_reward = -float('inf')

    print("Dummy Evaluation!!")
#     breakpoint()
    # evaluation은 항상 남양 트랙으로 수행 
    evaluate(args=args, env=nam_env, model=model, num_steps=10,
             save_frame=False, deterministic=True)
    print("Finished Dummy Evaluation!!")
    while num_episodes < MAX_EPISODES:
        if args.save is not None:
            # save_path = f"{os.path.dirname(os.path.abspath(__file__))}/experiments"
            save_path = exp_path
            # save_path = f"{save_path}/{args.save}"
            os.makedirs(save_path, exist_ok=True)
            model.save(f"{save_path}/{args.save}") ## save trained model (for the most optimal one) ##
            pickle.dump({'args': args.__dict__,
                         'obs': OBSERV_CONFIG}, 
                         open(f"{save_path}/args.pkl", "wb"))

        #total_timesteps: 하나의 episode에 할당하는 max timestep이라고 생각할 수 있음#
        if train_onload and num_episodes == 0: 
            model = model.learn(total_timesteps=args.total_timesteps, log_interval=10, progress_bar=True, 
                                reset_num_timesteps = False)
        else:
            model = model.learn(total_timesteps = args.total_timesteps, log_interval=10, progress_bar=True,
                                )
        ### evaluate on nam track for every episode ###
        print("=====evaluating======")
        # eval_model = copy.deepcopy(model) 
        reward_log, did_succeed, logger = evaluate(
            args=args,
            env=nam_env, model=model, 
            num_steps=int(1e+5),
            frame_save_range=10000, 
            deterministic=True, ##분명히 deterministic=True로 설정해 뒀는데, 궤적이 진동을 한다는 것은 ..
            save_frame=True
        ) #evaluate할 때마다 이미지 frame을 저장하기 때문에..#
        
        if did_succeed:
            to_log = logger.log_data
            to_log['reward'] = reward_log['full']
            log_root = f"{save_path}/nam_logs";os.makedirs(log_root, exist_ok=True)
            pickle.dump(to_log, open(f"{log_root}/episode_{num_episodes}.pkl", "wb"))
            print(f"Succeeded nam-c track on episode {num_episodes}")
            model_save_path = f"{save_path}/{num_episodes}" ## did_succeed하면 그냥 episode 번호로 저장!!
            model.save(model_save_path)
        else:
            print(f"Failed nam-c track on episode {num_episodes}")
        
        '''여기서 동일한 namn_env를 계속 사용하는데 reset할 때에 frames 배열은 reset이 안되었다.
        때문에 계속 배열에 array가 쌓이다보면 memory allocation error이 발생할 수 밖에 없는 것이다.'''
        if hasattr(nam_env, "frames"):
            nam_env.frames = []
        
        print("==========evaluation ended=============")   
        model.policy.set_training_mode(True)
        num_episodes += 1
        
        
        if args.n_envs > 0:
            val = vec_env.get_attr('num_episode')
            for i in range(args.n_envs):
                vec_env.set_attr('num_episode', val[i]+1, i)
        else:
            vec_env.num_episode += 1
        # if args.weight_save_converge:
        #     check_converge()
        if args.weight_save_interval > 0:
            if (num_episodes % args.weight_save_interval) == 0: ## 일정 episode interval 마다 저장!!
                model_save_path = f"{save_path}/{num_episodes}_log_weight"
                model.save(model_save_path)


        
        

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
         
        