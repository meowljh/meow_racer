'''
will be generating the trajectories from the RL agent's weights saved during training
based on the assumption that the longer the agent has been trained, the better policies the agent ought to have.
--> if the assumption is actually CORRECT, then the lap time must be optimized as training time passes.

[Settings]
1. logging dir (main folder): D:/rl_traj_agents
2. logging format: folder with the name of the experiment under the main folder

'''
import numpy as np
import pickle
import os, sys
from PIL import Image
from pathlib import Path
from glob import glob
from natsort import natsorted
from argparse import ArgumentParser
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root);sys.path.append(os.path.dirname(root))
from evaluate import evaluate
from envs.gym_car_racing import Toy_CarRacing
from envs.gym_car_racing_nam import Toy_CarRacing_NAM
from rl_src.stable.sac import SAC

TRAJ_LOG_PATH = "D:/rl_traj_agents"
TRAIN_EXP_PATH = f"{root}/experiments"

def int2bool(i):
    return True if i == 1 else False

class ARGS():
    def __init__(self, arg_dict):
        for key, value in arg_dict.items():
            setattr(self, key, value)
            
def get_args():
    parser = ArgumentParser()
    # parser.add_argument("--exp_name", type=str, required=True)
    
    parser.add_argument("--env_type", type=str, choices=["random", "nam"])
    parser.add_argument("--deterministic", type=int)
    parser.add_argument("--optimal", type=int, help="if set to 1, then we will be generating a single trajectory only with the optimal policy")
    parser.add_argument("--save_folder", type=str, default=None)
    parser.add_argument("--frame_save_range", type=int, default=50)
    parser.add_argument("--num_steps", type=int, default=int(1e+6))
    
    
    args = parser.parse_args()
    args.deterministic = int2bool(args.deterministic)
    args.optimal = int2bool(args.optimal)
    
    return args

def _prepare_all(exp_name, model_name, env):
    exp_path = os.path.join(TRAIN_EXP_PATH, exp_name)

    assert os.path.isdir(exp_path)
    model_path = f"{exp_path}/{model_name}.zip"
    
    agent = SAC(
        policy="MlpPolicy",
        env=env,
        action_noise=None,
        verbose=int(np.random.rand(1)*1000)
    )
    agent = agent.load(model_path)
    print(f"Successfully Loaded the Pre-Trained SAC for {model_name}")
    return agent

def run_and_make(agent, 
                 env, 
                 save_folder:str, 
                 model_name:str,
                 inp_args):
    log_folder = f"{TRAJ_LOG_PATH}/{save_folder}/{model_name}"
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    erg_log_path = f"{log_folder}/erg_log.pkl"
    if os.path.isfile(erg_log_path):
        return
    
    did_finish = False
    while not did_finish:
        reward_dict, did_finish, logger = evaluate(
            env=env,
            model=agent,
            num_steps=inp_args.num_steps,
            frame_save_range=inp_args.frame_save_range,
            deterministic=inp_args.deterministic
        )
        
        if (hasattr(env, 'frames') and did_finish) or (hasattr(env, 'frames') and inp_args.env_type == 'nam'):
            gif_paths = [] 
            image_log_path = f"{log_folder}/frames";os.makedirs(image_log_path, exist_ok=True)
    
            for fi, frame in enumerate(env.frames):
                gif_paths.append(f"{image_log_path}/{fi}.png")
                render = np.flip(frame, axis=0)
                render = np.rot90(render, k=3, axes=(0,1))
                render = render.astype(np.uint8)
                Image.fromarray(render).save(f"{image_log_path}/{fi}.png")

            im1 = Image.open(gif_paths[0])
            im1.save(f"{log_folder}/frames.gif", save_all=True, 
                     append_images=[Image.open(i) for i in gif_paths[1:]],
                     duration=100, loop=0)
    
    print(f"DID FINISH: {did_finish}")
     
    pickle.dump(logger.log_data, open(erg_log_path, 'wb'))
    
    reward_log_path = f"{log_folder}/reward_log.pkl"
    pickle.dump(reward_dict, open(reward_log_path, 'wb'))
    
    
if __name__ == "__main__":
    inp_args = get_args()
    EXP_NAME =  'BOTH_JWbaseline_0209_sac_3_5_15' # 'BOTH_ORGbaseline_0207_sac_mm01_tp05' # 'BOTH_JWbaseline_0209_sac_3_5_15'
    exp_path = f"{TRAIN_EXP_PATH}/{EXP_NAME}"
    config_path = f"{exp_path}/args.pkl"
    config = pickle.load(open(config_path, 'rb'))
    
    obs_conf = config['obs']
    env_conf = config['args']
    if 'use_theta_diff' not in env_conf:
        env_conf['use_theta_diff'] = False
    if inp_args.env_type == "nam":
        env = Toy_CarRacing_NAM(observation_config=obs_conf, lap_complete_percent=1, do_zoom=3., args=ARGS(env_conf))
    elif inp_args.env_type == "random":
        env = Toy_CarRacing(observation_config=obs_conf, lap_complete_percent=1., do_zoom=3., simple_circle_track=False, args=ARGS(env_conf))
    else:
        raise UserWarning(f"Unable to load environment for {inp_args.env_type}")
        
    if inp_args.optimal:
        agent = _prepare_all(EXP_NAME, model_name=EXP_NAME, env=env)
        run_and_make(agent=agent, env=env, 
             save_folder=inp_args.save_folder if inp_args.save_folder is not None else EXP_NAME,
             model_name=EXP_NAME,
             inp_args=inp_args)
        
    else:
        success_episodes = glob(exp_path + "/*.zip")
        success_episodes = [p for p in success_episodes if '_log_weight' not in p]
        success_episodes = natsorted(success_episodes)
        
        for ei, model_path in enumerate(success_episodes):
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            agent = _prepare_all(EXP_NAME, model_name=model_name, env=env)
            
            run_and_make(agent=agent, env=env, 
                         save_folder=inp_args.save_folder if inp_args.save_folder is not None else EXP_NAME,
                         model_name=model_name,
                         inp_args=inp_args)
            
    
        
        