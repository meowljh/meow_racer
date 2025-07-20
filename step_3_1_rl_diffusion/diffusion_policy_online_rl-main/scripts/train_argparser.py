import argparse

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", type=str, default="sdac")
    parser.add_argument("--env", type=str, default="MountainCarContinuous-v0")
    parser.add_argument("--suffix", type=str, default="gym_classic_control")
    parser.add_argument("--render_mode", type=str, default="human")
    parser.add_argument("--num_vec_envs", type=int, default=1)
    parser.add_argument("--hidden_num", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--diffusion_steps", type=int, default=20)
    parser.add_argument("--diffusion_hidden_dim", type=int, default=256)
    parser.add_argument("--start_step", type=int, default=int(3e4))
    parser.add_argument("--total_step", type=int, default=int(1e6))
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_schedule_end", type=float, default=3e-5)
    parser.add_argument("--alpha_lr", type=float, default=7e-3)
    parser.add_argument("--delay_alpha_update", type=float, default=250)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--num_particles", type=int, default=20)
    parser.add_argument("--noise_scale", type=float, default=0.1)
    parser.add_argument("--cluster", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--beta_schedule_scale", type=float, default=0.8)
    parser.add_argument("--beta_schedule_type", type=str, default="linear")
    #### for SDAC ####
    parser.add_argument("--time_dim", type=int, default=16)
    parser.add_argument("--target_entropy_scale", type=float, default=0.9)
    parser.add_argument("--activation_fn", type=str, default='mish')
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--delay_update", type=int, default=2)
    parser.add_argument("--reward_scale", type=float, default=0.2)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--reverse_mc_num", type=int, default=64)
    
    #### for trainer setup ####
    parser.add_argument("--sample_per_iteration", type=int, default=1)
    parser.add_argument("--update_per_iteration", type=int, default=1)
    parser.add_argument("--evaluate_n_episode", type=int, default=20)
    parser.add_argument("--evaluate_every", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)

    args = parser.parse_args()

    return args
