import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
import os, sys
import threading, time, pygame
from collections import defaultdict
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root);sys.path.append(os.path.dirname(root))

from envs.gym_car_constants import *
from envs.gym_car_racing import Toy_CarRacing 
from envs.utils import calculate_error_c, make_gif, key_press_check

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control_steer", action="store_true")
    parser.add_argument("--control_gas", action="store_true")
    parser.add_argument("--target_vel", type=float, default=50) # m/s to match the gym CarRacing env #
    
    parser.add_argument("--steer_p", type=float, default=1 )
    parser.add_argument("--steer_i", type=float, default=0.)
    parser.add_argument("--steer_d", type=float, default=0.)
    
    parser.add_argument("--gas_p", type=float, default=1.)
    parser.add_argument("--gas_i", type=float, default=0.)
    parser.add_argument("--gas_d", type=float, default=0.)
    
    parser.add_argument("--num_runs", type=int, default=50)
    parser.add_argument("--log_path", type=str, required=True)
    
    args = parser.parse_args()
    
    
    # args.gas_p =0.1
    # args.gas_i = args.gas_p 
    
    # args.steer_d = args.steer_p * 2

    # args.gas_i = args.gas_p * 5
    # args.gas_d = args.gas_p * 10
    
    return args

class PID_Controller(object):
    def __init__(self,
                 dt,
                 steer_p:float, gas_p:float,
                 steer_i:float=0., gas_i:float=0.,
                 steer_d:float=0., gas_d:float=0.):
        super(PID_Controller, self).__init__()
        self.dt = dt
        self.steer_p = steer_p
        self.steer_i = steer_i
        self.steer_d = steer_d
        self.gas_p = gas_p
        self.gas_i = gas_i
        self.gas_d = gas_d
        
        self.vel_error = [0]
        self.pos_error = [0]
        
    def reset(self):
        self.vel_error = [0]
        self.pos_error = [0]
        
    def gas_control(self, vel, target_vel):
        control = 0
        err = target_vel - vel 
        control = self.gas_p * err
        if self.gas_d != 0: 
            control += self.gas_d * (err - self.vel_error[-1]) / self.dt
        if self.gas_i != 0:
            control += self.gas_i * np.sum(self.vel_error) * self.dt
        
        self.vel_error.append(err)
        
        return control
    
    def steer_control(self, center_dist):
        # center_dist, _, _, ref = calculate_error_c(x, y, track_x=cx_arr, track_y=cy_arr, phi_arr=phi_arr, 
        #                                 kappa_arr=kappa_arr, debug=True)
        # breakpoint()
        
        target_dist = 0.
        err = target_dist - center_dist
        control = self.steer_p * err # p제어 #
        if self.steer_d != 0:
            control += self.steer_d * (err - self.pos_error[-1]) / self.dt # d제어 #
        if self.steer_i != 0:
            control += self.steer_i * np.sum(self.pos_error) * self.dt # i제어 #
            
        self.pos_error.append(err)
        
        # return control 
        return -control
            
def load_thread(status_arr):
    tr = threading.Thread(target=key_press_check, name='check', args=(status_arr,))
    tr.daemon = True 
    tr.start()
    return tr

def main_control(env, controller,  pid_args):
    running_rewards = []
    env.reset()
    
    root = os.path.dirname(os.path.abspath(__file__))
    
    run_log_path = f"{root}/{pid_args.log_path}";os.makedirs(run_log_path, exist_ok=True)
    
    for run in range(pid_args.num_runs):
        done = False
        observations = env.reset()
        reward_sum = 0
        repeat_neg_reward = 0
        
        time_iter = 0
        
        car_y_arr = []
        car_x_arr = []
        steer_arr = []
        gas_arr = []
        center_dist_arr = []
        ref_x_arr, ref_y_arr = [], []
        
        angle_arr = []
        
        track_x = env.track_dict['x']
        track_y = env.track_dict['y']
        
        # status_list = []
        status_dict = defaultdict(int)
        prev_status = 1
        key_thread = load_thread(status_arr=status_dict)
 
    
        while not done: 
            state = observations[0]   
            status = env._track_events(status_dict)  
            prev_status = status if status is not None else prev_status
            # print(status, prev_status, status_dict)
    
            if status == -1:
                env._quit()
            
            
            elif (status == 0) or (status is None and prev_status == 0): #pause
                prev_status = 0
                env._pause() 
                key_thread.join() 
                status_dict = defaultdict(int) 
                ## pause 누르고, thread 새로 만들어서 다시 tracking 시작 ##
                key_thread = load_thread(status_arr=status_dict)  
                
                while True:  
                    status = env._track_events(status_dict) 
                    prev_status = status if status is not None else prev_status
 
                    if status == 1:  
                        prev_status = 1
                        break 
                    
                    elif status == -1: 
                        env.quit()
                        
                    elif status == 0: ## 또 PAUSE를 누른 경우 ##   
                        status_dict = defaultdict(int) ## 유효한 keyboard를 누른 것이기 때문에 keyboard tracker thread를 새로 만들어야 함 ##
                        key_thread = load_thread(status_arr=status_dict)
                        
                    elif status is None: 
                        status = 0
                    
                    key_thread.join()
             
                status_dict = defaultdict(int)
                key_thread = load_thread(status_arr=status_dict)
            
            else:  
                car_x, car_y = env.car.hull.position 
                # if car_y > 50:
                #     breakpoint()
                car_x_arr.append(car_x)
                car_y_arr.append(car_y)

                car_vel = np.mean(np.array([w.v for w in env.car.wheels]), 0)
                car_vel = math.sqrt(car_vel[0]**2 + car_vel[1]**2) ## 이 부분에서 그냥 차량의 속도를 잘못 계산했던게 문제였음 ##

                center_dist, _, _, ref = calculate_error_c(x=car_x,
                                                    y=car_y,
                                                    track_x=track_x,
                                                    track_y=track_y,
                                                    phi_arr=env.track_dict['phi'],
                                                    kappa_arr=env.track_dict['kappa'],
                                                    debug=True
                                                    )
                # center_dist =  -math.sin(ref[2])*(car_x-ref[0]) +   math.cos(ref[2])*(car_y-ref[1])
                # center_dist = float(center_dist)

                ref_x_arr.append(ref[0])
                ref_y_arr.append(ref[1])
                angle_arr.append(env.car.hull.angle)
                # if car_x < -10:
                #     breakpoint()
                steer_control = controller.steer_control(center_dist)

                steer_arr.append(steer_control)
                center_dist_arr.append(center_dist)

                gas_control = controller.gas_control(vel=car_vel, target_vel=pid_args.target_vel)
                gas_arr.append(gas_control)

                if gas_control < 0:
                    action = [steer_control, 0, -gas_control]
                else:
                    action = [steer_control, gas_control, 0.] 

                observations = env.step(action=action)
    
                reward = observations[1]
                reward_sum += reward
                repeat_neg_reward = repeat_neg_reward + 1 if reward < 0 else 0
                done = observations[-2]

                if time_iter % FPS == 0:
                    env._save_frames(fname = f"{run_log_path}/{time_iter}.png")

                if time_iter % 100 == 0 or done:
                    fig, ax = plt.subplots(figsize=(4,6))
                    plt.scatter(env.track_dict['x'], env.track_dict['y'], s=1, alpha=0.2, c='k')
                    plt.scatter(car_x_arr, car_y_arr, c='r', s=2)
                    plt.savefig(f"{run_log_path}/track_route.jpg")
                    plt.close()
                    fig, ax = plt.subplots(3, 1, figsize=(7,5))
                    ax[0].plot(steer_arr, c='k');ax[0].set_title("$ steer $")
                    ax[2].plot(center_dist_arr, c='k');ax[2].set_title("$ center_dist $")
                    ax[1].plot(gas_arr, c='k');ax[1].set_title("$ gas $")
                    plt.savefig(f"{run_log_path}/pid_control.jpg")
                    plt.close()
                    import pickle
                    pickle.dump({'car_x': car_x_arr, 'car_y': car_y_arr, 'track_x': env.track_dict['x'],
                                 'track_y': env.track_dict['y'],
                                 'center_dist': center_dist_arr,
                                'ref_x': ref_x_arr,
                                'ref_y': ref_y_arr,
                                'angle': angle_arr},
                                open(f'{run_log_path}/pid_run_log.pkl', 'wb'))
                time_iter += 1 
                if repeat_neg_reward >= 300:
                    break 
            
        running_rewards.append(reward_sum)
        make_gif(image_root=run_log_path, dest_path=f"{run_log_path}/run.gif")
        pickle.dump(
            env.actor_state_dict,
            open(f'{run_log_path}/actor_state.pkl', 'wb')
        )
    
if __name__ == "__main__":
    '''PID 제어
    (1) steer control
        - 조향각을 조절하는 관점이다.
        - 따라서 center line을 따라가도록, 그 오차로 PID 제어를 하면 steering을 조절할 수 있다.
        
    (2) gas control
        - 정속 주행을, 즉 정해진 속도로 차량이 주행을 하도록 하면 그 오차를 통해서 gas를 조절할 수 있다.
        
    (3) brake control 
        - PID제어로 gas와 brake를 모두 다루기는 까다로울 수 있음. 
        - 일방적으로 0으로 두거나 속도 차이가 음수일때 (줄여야할 때) brake로 갈아타는 방법도 있음.
    '''
    
    observation_config = {
        'dynamic': ['theta', 'e_c', 'e_phi', 'v_x', 'v_y', 'omega']
    }
    env = Toy_CarRacing(observation_config=observation_config, simple_circle_track=True,
                        do_zoom=False)
    args = load_args()
    
    controller = PID_Controller(dt=1/FPS,
                                steer_p=args.steer_p, steer_i=args.steer_i, steer_d=args.steer_d,
                                gas_p=args.gas_p, gas_i=args.gas_i, gas_d=args.gas_d)
    running_rewards = []
    runs = 50
    for run in range(runs):
        
        repeat_neg_reward = 0
        done = False
        observations = env.reset()
        reward_sum = 0
        
        while not done: 
            state = observations[0]
            car_x, car_y = env.car.hull.position 
            car_vel = np.mean(np.array([w.v for w in env.car.wheels]), 0)
            car_vel = math.sqrt(car_vel[0]**2 + car_vel[1]**2) ## 이 부분에서 그냥 차량의 속도를 잘못 계산했던게 문제였음 ##
        
            if args.control_steer:
                steer_control = controller.steer_control(x=car_x, y=car_y,
                                                         cx_arr=env.track_dict['x'], cy_arr=env.track_dict['y'],
                                                         phi_arr=env.track_dict['phi'],
                                                         kappa_arr=env.track_dict['kappa'])
            if args.control_gas:
                gas_control = controller.gas_control(vel=car_vel, target_vel=args.target_vel)
           
                if gas_control < 0:
                    action = [steer_control, 0, -gas_control]
                else:
                    action = [steer_control, gas_control, 0.] 
             
            observations = env.step(action=tuple(action))
 
            reward = observations[1]
            reward_sum += reward
            repeat_neg_reward = repeat_neg_reward + 1 if reward < 0 else 0
            if repeat_neg_reward >= 300:
                break
        running_rewards.append(reward_sum)    
            
        