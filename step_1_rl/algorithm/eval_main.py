import argparse
import gymnasium as gym

from ppo.ppo_algorithm import PPO
from trpo.trpo_algorithm import TRPO
from trpo.reinforce_algorithm import Reinforce
from trpo.reinforce_baseline_algorithm import Reinforce_Baseline

def load_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--rl_name", type=str, required=True)
    parser.add_argument("--env_name", type=str, required=True)

    args = parser.parse_args()
    
    return args


def load_env(args):
    ##MuJoCo##
    if args.env_name.lower() == "cheetah":
        return gym.make("HalfCheetah-v5",  render_mode="human") # xml_file=args.xml_file) #
    elif args.env_name.lower() == "humanoid":
        return gym.make("Humanoid-v4", render_mode="human") # xml_file=args.xml_file) #
    elif args.env_name.lower() == "humanoid_standup":
        return gym.make("HumanoidStandup-v4",  render_mode="human") # xml_file=args.xml_file) #
    elif args.env_name.lower() == "ant":
        return gym.make('Ant-v4', render_mode="human") #xml_file=args.xml_file)
    elif args.env_name.lower() == "hopper":
        return gym.make("Hopper-v5", render_mode="human") # , xml_file=args.xml_file)
    elif args.env_name.lower() == "swimmer":
        return gym.make("Swimmer-v2", render_mode="human")  # xml_file=args.xml_file)
    elif args.env_name.lower() == "walker":
        return gym.make('Walker2d-v4', render_mode="human") #  xml_file=args.xml_file)
    elif args.env_name.lower() == "reacher":
        return gym.make("Reacher-v4", render_mode="human") #  xml_file=args.xml_file)
    ##Box2D##
    elif args.env_name.lower() == "mountain_car":
        return gym.make("MountainCarContinuous-v0", render_mode="human")
    ##Classic Control##
    elif args.env_name.lower() == "bipedal":
        return gym.make("BipedalWalker-v3", hardcore=True)
    elif args.env_name.lower() == "pendulum":
        return gym.make("Pendulum-v0", render_mode="human")
    

def load_agents(args):
    if args.rl_name.upper() == "PPO":
        agent = PPO()

    elif args.rl_name.upper() == "TRPO":
        agent = TRPO()
        
    elif args.rl_name.upper() == "REINFORCE":
        agent = Reinforce()

    elif args.rl_name.upper() == "REINFORCE_BASE":
        agent = Reinforce_Baseline()
        
    elif args.rl_name.upper() == "TD3":
        pass
    elif args.rl_name.upper() == "SAC":
        pass
    elif args.rl_name.upper() == "DDPG":
        pass
    else:
        raise UserWarning(f"Undefined RL Agent:  {args.rl_name.upper()}")
    
    return agent

if __name__ == "__main__":
    args = load_args()