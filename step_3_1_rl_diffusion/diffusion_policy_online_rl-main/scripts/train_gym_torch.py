"""train_classic_control_torch.py
[Supporting Environments]
- Mountain Car Continuous
- Pendulum
"""

import argparse
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 

sys.path.append(ROOT)
sys.path.append(f"{ROOT}/meow")

from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent #step_3_1_rl_diffusion
 
import time
from functools import partial
import yaml


import torch
import torch.nn as nn

##algorithm##
from meow.torch_algorithm.sac import SAC
from meow.torch_algorithm.sdac import SDAC
##network##
from meow.torch_network.sac import SACNet
from meow.torch_network.diffv2 import Diffv2Net

from meow.torch_buffer.replay_buffer import ReplayBuffer
from meow.torch_trainer.off_policy import OffPolicyTrainer
from meow.torch_utils.experience import Experience
from meow.torch_utils.seeding import seed_all
from meow.torch_env.build_env import build_gym_env

from train_argparser import get_argparser

if __name__ == "__main__":
    args = get_argparser()

    master_seed = args.seed
    seed_all(seed=master_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_env, obs_dim, act_dim = build_gym_env(env_name=args.env, render_mode=args.render_mode, seed=master_seed)
    eval_env = None

    hidden_sizes = [args.hidden_dim] * args.hidden_num
    diffusion_hidden_sizes = [args.diffusion_hidden_dim] * args.hidden_num

    replay_buffer = ReplayBuffer(max_len=int(1e6), obs_dim=obs_dim, action_dim=act_dim, horizon_len=None)
    
    if args.alg == 'sac':
        net = SACNet(device=device, obs_dim=obs_dim, act_dim=act_dim, style_dim=0, hidden_sizes=hidden_sizes, activation_fn='gelu')
        algorithm = SAC(device=device, agent=net, lr=args.lr)
    
    elif args.alg == 'sdac':
        target_entropy = -act_dim * args.target_entropy_scale
        net = Diffv2Net(device=device, num_timesteps=args.diffusion_steps, time_dim=args.time_dim,
                        act_dim=act_dim, obs_dim=obs_dim, style_dim=0, hidden_sizes=hidden_sizes,
                        num_particles=args.num_particles, target_entropy=target_entropy,
                        beta_schedule_scale=args.beta_schedule_scale, activation_fn=args.activation_fn,
                        output_activation_fn='identity', beta_schedule_type=args.beta_schedule_type,
                        noise_scale=args.noise_scale)
        algorithm = SDAC(device=device, agent=net, gamma=args.gamma, lr=args.lr, alpha_lr=args.alpha_lr,
                         lr_schedule_end=args.lr_schedule_end, tau=args.tau, 
                         delay_alpha_update=args.delay_alpha_update,
                         delay_update=args.delay_update,
                         reward_scale=args.reward_scale, num_samples=args.num_samples,
                         reverse_mc_num=args.reverse_mc_num)

    else:
        raise NotImplementedError(f"algorithm {args.alg} not implemented")
    
    exp_dir = PROJECT_ROOT / "logs" / args.env / (args.alg + '_' + time.strftime("%Y-%m-%d_%H-%M-%S") + f"_s{args.seed}_{args.suffix}")
    
    trainer = OffPolicyTrainer(
        seed=master_seed,
        device=device,
        env=train_env,
        algorithm=algorithm,
        buffer=replay_buffer,
        log_path=exp_dir,
        batch_size=args.batch_size,
        start_step=args.start_step,
        total_step=args.total_step,
        sample_per_iteration=args.sample_per_iteration,
        update_per_iteration=args.update_per_iteration,
        evaluate_every=args.evaluate_every,
        evaluate_n_episode=args.evaluate_n_episode,
        evaluate_env=eval_env,
        save_policy_every=int(args.total_step / 20),
        warmup_with="random", 
        update_log_n_step=1 if args.debug else 1000
    )

    trainer.setup(Experience.create_example(obs_dim=obs_dim, action_dim=act_dim, batch_size=trainer.batch_size, horizon_size=None))

    # save the arguments to a YAML file #
    args_dict = vars(args)
    with open(os.path.join(exp_dir, "config.yaml"), "w") as yf:
        yaml.dump(args_dict, yf)

    trainer.run()
    
    


    



