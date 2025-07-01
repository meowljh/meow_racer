import os,sys
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root);sys.path.append(os.path.dirname(root))
from PIL import Image
import numpy as np
import torch
import argparse
 
import pickle
sys.path.append(f"{root}/rl_src")
from evaluate import evaluate
from stable.sac import SAC
from stable.common.noise import NormalActionNoise

from step_1_rl.envs.gym_car_racing_nam import Toy_CarRacing_NAM
from step_1_rl.envs.gym_car_racing import Toy_CarRacing

from step_1_rl.tutorial_train import INP_ARGS
 

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deterministic", type=int, default=1)
    parser.add_argument("--env_type", type=str, choices=["nam", "random"])
    parser.add_argument("--folder_name", type=str, default="None")
    parser.add_argument("--episode_num", type=int, default=-1)
    
    args = parser.parse_args()
    args.folder_name = None if args.folder_name == 'None' else args.folder_name
    args.deterministic = True if args.deterministic == 1 else False
    
    return args



def update_obs_conf(obs_conf, args):
    if args.use_brake:
        obs_conf['car'].append("brake")
    if args.use_gas:
        obs_conf['car'].append("gas")
    if args.use_steer:
        obs_conf['car'].append("steer")
    if args.use_delta:
        obs_conf['car'].append("delta")
    if args.use_force:
        # obs_conf['car'].extend([f"force_{i}" for i in range(4)])
        obs_conf['car'].append("force")
    return obs_conf

args = INP_ARGS()
args.same_tile_penalty = 0.
args.min_vel = 0.
args.time_penalty = 0.5
args.min_movement = 0.1
args.reward_type = "baseline"
args.use_gas = False  # True
args.use_steer = False # True
args.use_brake = False  #True
args.use_delta = False  #True
args.use_force =False #  True


args.frame_save_range = 25


if __name__ == "__main__":
    user_args = load_args()
    EXP_ROOT_PATH = f"{os.path.dirname(os.path.abspath(__file__))}/experiments"
    
    # EXP_NAME = "CENTER_BOTH_RAND4_REV_RANDSTART_JW_0227_3515"
    # EXP_NAME = "BOTH_RAND4_JWbaseline_0301_PENCHANGE" # "OSCI_NEWPEN2_RAND4_ONLYENV_REV_JWbaseline_0307_3_5_15" #  "BOTH_baseline_0203_sac_mm01_tp05" # "baseline_0114_sac_mm01_tp05" # "baseline_0114_sac_mm01_tp05" #"sac_baseline_tp_5 mm_1 8vec" # "baseline tp_5 mm_1" # 
    EXP_NAME = 'JW_ENVONLY_BOTH_RANDSTART_REV_0224_3515'
    ENV_NAME = user_args.env_type #   "nam" # "random" # "nam" 
    IS_DETERMINISTIC = user_args.deterministic
    
    arg_path = os.path.join(EXP_ROOT_PATH, EXP_NAME, 'args.pkl')
    exp_args_saved = pickle.load(open(arg_path, 'rb'))['args']
    for key, value in args.__dict__.items():
        if key in exp_args_saved:
            # args[key] = exp_args_saved[key]
            setattr(args, key, exp_args_saved[key])

    obs_conf = {
    'dynamic': ['e_c', 'e_phi', 'v_x', 'v_y'],
    'lidar': [f"lidar_{i}" for i in range(int(180 / args.lidar_deg))],
    'car': [f"omega_{i}" for i in range(4)] + \
            [f"forward_{i}" for i in range(2*args.num_vecs)]
    }
    obs_conf = update_obs_conf(obs_conf, args)


    if ENV_NAME == "random":
        env = Toy_CarRacing(observation_config=obs_conf, do_zoom=3., args=args )
    elif ENV_NAME == "nam":
        env = Toy_CarRacing_NAM(observation_config=obs_conf, do_zoom=3., args=args)
    else:
        raise UserWarning(f"Environment {ENV_NAME} not supported..")
    

    config = pickle.load(open(f"experiments/{EXP_NAME}/args.pkl", "rb"))
    if user_args.episode_num == -1:
        model_path = f"experiments/{EXP_NAME}/{EXP_NAME}.zip"
    else:
        model_path = f"experiments/{EXP_NAME}/{user_args.episode_num}.zip"
    
    model = SAC(
        policy="MlpPolicy",
        env=env,
        action_noise=None, # no action noise for evaluation mode #
        verbose=1,
        seed=int(np.random.rand(1) * 1000,),
    )
    model = model.load(model_path)
    print("MODEL SUCCESSFULLY LOADED!!")
 
    print("STARTING EVALUATION!!")

    did_finish = False
    while did_finish is False:
        reward_dict, did_finish, logger = evaluate(env=env, 
                                                   model=model, 
                                                   num_steps=int(1e+6), 
                                                   frame_save_range=args.frame_save_range,
                                                   deterministic=IS_DETERMINISTIC)
        
        if (hasattr(env, 'frames') and did_finish) or (hasattr(env, 'frames') and ENV_NAME == 'nam'):
            gif_paths = []
            if user_args.folder_name is None:
                folder = f"experiments/{EXP_NAME}/render_{ENV_NAME}";os.makedirs(folder, exist_ok=True)
            else:
                folder = f"experiments/{EXP_NAME}/render_{user_args.folder_name}";os.makedirs(folder, exist_ok=True)
            # pickle.dump(env.frames, open(f"{folder}/out.pkl", "wb"))
            # frames = pickle.load(open(f"{folder}/out.pkl", "rb"))
            # for fi, frame in enumerate(frames):
            for fi, frame in enumerate(env.frames):
                gif_paths.append(f"{folder}/{fi}.png")
                render = np.flip(frame, axis=0)
                render = np.rot90(render, k=3, axes=(0,1))
                render = render.astype(np.uint8)
                Image.fromarray(render).save(f"{folder}/{fi}.png")

            im1 = Image.open(gif_paths[0])
            im1.save(f"{folder}/out.gif", save_all=True, 
                     append_images=[Image.open(i) for i in gif_paths[1:]],
                     duration=250,  # 프레임 전환 속도 -> 0.1초 #
                     loop=5 # 전체 반복 횟수
                     )
            if ENV_NAME == 'nam':
                did_finish=True
                
            
        # for key, value in reward_dict.items():
        #     print(f"{key} ---> {value}")
        print(f"Finished:  {did_finish}")
        log_path = f"{folder}/erg_log.pkl"
        pickle.dump(logger.log_data, open(log_path, 'wb'))
        reward_path = f"{folder}/reward_log.pkl"
        pickle.dump(reward_dict, open(reward_path, 'wb'))