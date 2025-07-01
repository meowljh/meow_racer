import torch
from data_logger import ERGLogger
import numpy as np
from gymnasium import spaces
import math

@torch.no_grad()
def evaluate(args,
             env, model, num_steps,  
             deterministic:bool=True,
             frame_save_range:int=20,
             save_frame:bool=False,
             early_stop:bool=False):
    obs, _ = env.reset()
    logger = ERGLogger(car_obj=env.car, env=env)
    
    running_reward = {'step':[], 'full': []}
    model.policy.set_training_mode(False) # actor, critic 모두 training_mode=False로 설정 #

    state = None
    truncated = False
    terminated = False 
    '''혹시나 action_noise가 training_mode=False로 뒀을 때에도 적용이 되나 걱정했었는데, 그건 아닌 것 같고 sample_action 함수 부분에서 봤을 때
    collect_rollout단에서 action_noise가 사용되고 있음을 알 수 있다.
    그래서 이건 학습 중에 Replay Buffer에 past experience data를 저장하기 위한 용도이기 때문에, 그 이슈는 아닌듯 하다'''

    no_move = 0
    cx, cy = env.car.hull.position
    
    for step in range(1, num_steps+1):
        if isinstance(obs, dict):
            # obs = obs['obs']
            obs['obs'] = torch.tensor(obs['obs'], device=model.device).reshape(1, -1)
        else:
            obs = torch.tensor(obs, device=model.device).reshape(1, -1)
        
        with torch.no_grad():
            if args.rl_algorithm.upper() == "PPO":
                actions = model.policy._predict(obs, deterministic=deterministic)
            elif args.rl_algorithm.upper() == "SAC":
                # actions_pi, _ = model.actor.action_log_prob(obs)
                # actions = model.actor._predict(obs, deterministic=True)
            
                actions = model.actor._predict(obs, deterministic=deterministic)       
        

        if args.rl_algorithm.upper() == "SAC":
            actions = actions.cpu().numpy().reshape((-1, *model.actor.action_space.shape))
            
            if isinstance(model.actor.action_space, spaces.Box):
                if model.actor.squash_output: #default is TRUE for SAC#
                    actions = model.actor.unscale_action(actions)
                else: 
                    actions = np.clip(actions, model.actor.action_space.low, model.actor.action_space.high)
        elif args.rl_algorithm.upper() == "PPO":
            actions = actions.cpu().numpy().reshape((-1, *model.policy.action_space.shape))
            
            if isinstance(model.policy.action_space, spaces.Box):
                if model.policy.squash_output:
                    actions = model.policy.unscale_action(actions)
                else:
                    actions = np.clip(actions, model.policy.action_space.low, model.policy.action_space.high)
                    
            # log_prob = log_prob.reshape(-1, 1)
            # breakpoint()
        # action, _ = model.policy.predict(
        #     observation=obs, state=state, episode_start=None, deterministic=False
        # )
        # action = actions_pi.detach().cpu().numpy().squeeze()
        actions = actions.squeeze()
        obs, reward, terminated, truncated, _ = env.step(actions)
        
        car_x, car_y = env.car.hull.position
        if math.sqrt((car_x-cx)**2+(car_y-cy)**2) < 0.1:
            no_move += 1
        else:
            no_move = 0
            
        if no_move > 100 and early_stop:
            terminated = True
        
        
        cx, cy = car_x, car_y
        if (step % frame_save_range) == 0:
            env._save_frame()
            
        running_reward['step'].append(reward)
        running_reward['full'].append(env.reward)
        
        logger.log_single_step(action=actions)
    ###############################
    ### 단순히 시간이 다 되어서 terminate되면 False이고, 성공적으로
    # 트랙을 완주한 경우에만 did_success=True 반환
    ###############################
        if terminated:
            return running_reward, False, logger
        elif truncated:
            return running_reward, True, logger
    
    return running_reward, False, logger