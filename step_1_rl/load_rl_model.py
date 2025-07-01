import os, sys
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)

import numpy as np

from rl_src.stable.td3 import TD3
from rl_src.stable.sac import SAC
from rl_src.stable.ppo import PPO

from rl_src.stable.common.nstep_buffers import NStepReplayBuffer
from rl_src.stable.common.lambda_step_buffers import LambdaStepReplayBuffer
from rl_src.stable.her.her_replay_buffer import HerReplayBuffer

from rl_src.stable.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, Linear_NormalActionNoise

def get_action_noise(args, vec_env, action_noise_type:str='None'):
    n_actions = vec_env.action_space.shape[-1]
    if args.action_noise_type.lower() == "gaussian" or action_noise_type.lower() == "gaussian":
        action_noise = NormalActionNoise(
            mean = np.zeros(n_actions),
            sigma = args.action_noise_sigma * np.ones(n_actions)
        )
    elif args.action_noise_type.lower() == "linear_gaussian" or action_noise_type.lower() == "linear_gaussian":
        action_noise = Linear_NormalActionNoise(
            mean=np.zeros_like(n_actions),
            sigma = args.action_noise_sigma * np.ones(n_actions),
            max_steps = args.max_gaussian_step,
            final_sigma = None if args.final_sigma == -1 else args.final_sigma * np.ones(n_actions),
        )
    elif args.action_noise_type.lower() == "brownian" or action_noise_type.lower() == "brownian": # 더 연속적인 action noise를 위해서 사용됨
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean = np.zeros(n_actions), # 평균
            sigma = args.action_noise_sigma * np.ones(n_actions), # 변동성
            theta = args.action_noise_theta, # 평균 회귀 계수
            dt = args.action_noise_dt, # 시간 간격 (환경의 dt와 보통 동일하게 설정)
        )
    elif args.action_noise_type.lower() == "none":
        action_noise = None
    
    return action_noise

def get_replay_buffer_class(args, vec_env):
    replay_buffer_kwargs = None
    if args.n_steps_td > 0:
        if args.step_lambda > 0:
            replay_buffer_class = LambdaStepReplayBuffer
            
        else:
            replay_buffer_class = NStepReplayBuffer
    
    elif args.replay_buffer_class == 'HerReplayBuffer':
        replay_buffer_class = None
        replay_buffer_kwargs = {'env': vec_env}
    else:
        replay_buffer_class = None
    
    return replay_buffer_class, replay_buffer_kwargs


def load_rl_algo(args, vec_env):
    policy_type = "MultiInputPolicy" if args.replay_buffer_class == "HerReplayBuffer" else "MlpPolicy"
    action_noise = get_action_noise(args=args, vec_env=vec_env)
    replay_buffer_class, replay_buffer_kwargs = get_replay_buffer_class(args=args, vec_env=vec_env)
    
    if args.rl_algorithm.upper() == "SAC": ## off-policy -> replay buffer needed
        model = SAC(
            policy=policy_type,
            env=vec_env,
            tau=args.tau,
            action_noise=action_noise,
            buffer_size=args.replay_buffer_size,
            batch_size=args.batch_size,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            ent_coef=args.ent_coef, # entropy, 즉 -pi(a_t|s_t) * log(pi(a_t|s_t))에 곱하는 weight로, entropy maximization을 위해 사용. 근데 target_entropy가 있고 이 값과 가까워지도록 objective function이 설계되어 있음
            gamma=args.gamma,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_step,
            learning_starts=args.learning_start,
            learning_rate=args.learning_rate,
            use_sde=False,
            ##### 사실 N-step TD를 SAC에 구현을 해 놓기는 했지만, 결과적으로는 off policy이기 때문에 오히려 다른 policy로 학습하는 점에 있어서의 bias가 점점 누적되어 가는 듯함. #####
            n_steps_td=args.n_steps_td,
            step_lambda=args.step_lambda,
            step_lambda_mode=args.step_lambda_mode,
            seed=args.random_seed
        )
    elif args.rl_algorithm.upper() == "PPO": ## on-policy -> no replay buffer needed
        model = PPO(
            policy=policy_type,
            env=vec_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps, # number of steps to run for each environment before the update - on policy 알고리즘이기 때문에 n_step만큼의 rollout data를 모아서 이걸로 학습을 시켜야 함.
            batch_size=args.batch_size,
            n_epochs=args.n_epochs, # number of epochs when optimizing the surrogate loss
            gamma=args.gamma, # discount factor (이후 time step의 value function에 대해서 할인율 부여)
            gae_lambda=args.gae_lambda, # factor for tradeoff of bias vs variance
            clip_range=args.clip_range, # clipping range for the advantage
            clip_range_vf=args.clip_range_vf, # clipping range for the value function
            normalize_advantage=args.normalize_advantage,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            use_sde=False, sde_sample_freq=-1,
            rollout_buffer_class=None, rollout_buffer_kwargs=None,
            target_kl=args.target_kl, # if use KL Divergence as loss to put the limit on the policy update, we need this. (default is None)
            verbose=1,
            seed=args.random_seed,
            use_beta_dist=args.use_beta_dist
        )
    
    return model
            
