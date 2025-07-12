import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
from environment.vehicle import RaceCar
from environment.track import Bezier_TrackGenerator, Nam_TrackGenerator

root=os.path.dirname(os.path.abspath(__file__))
#vehicle params
ACTION_DIM=3
DT=0.016 #0.2
CFG_FILE_PATH=f"{root}/environment/vehicle/jw_config.yaml"

#bezier track params
MIN_NUM_CKPT=4
MAX_NUM_CKPT=16
MIN_KAPPA=0.04
MAX_KAPPA=0.1
TRACK_WIDTH=7.

#nam track params
NAM_TRACK_PATH=f"{root}/statics/nam_c_track.pkl"

#sinusoidal control signal params
SIM_TIME=1000 #sec
FREQ=10 #0.1


################################################################################################
car = RaceCar(
    action_dim=ACTION_DIM,
    dt=DT,
    cfg_file_path=CFG_FILE_PATH
)

bezier_track_gen = Bezier_TrackGenerator(
    min_num_ckpt=MIN_NUM_CKPT, 
    max_num_ckpt=MAX_NUM_CKPT
)

nam_track_gen = Nam_TrackGenerator(
    track_width=TRACK_WIDTH,
    nam_track_path=NAM_TRACK_PATH,
    min_num_ckpt=MIN_NUM_CKPT, 
    max_num_ckpt=MAX_NUM_CKPT
)

bezier_track_gen._generate()
bezier_track_dict = bezier_track_gen._calculate_track_dict()

nam_track_gen._generate()
nam_track_dict = nam_track_gen._calculate_track_dict()

car._reset(nam_track_dict)


#begin dummy simulation
steer_action = 0. # car.bicycle_model.car_phi # 0. #조향은 하지 않는 상태로 고정.
brake_action = -1. #BPS도 밟지 않는 상태로 고정. (-1 ~ 1을 0 ~ 1 사이의 값으로 매핑하기 때문에 BPS=0인 것을 원한다면 -1로 action을 설정해야 함)

time = 0
action_dict = defaultdict(list)

for t in range(SIM_TIME):
    time += DT*t
    throttle_action = np.sin(2 * np.pi * FREQ * time)

    # brake_action = np.sin(2 * np.pi * FREQ * time)
    # brake_action = brake_action * 0.1 if brake_action > 0 else brake_action
    
    
    car._step(action=[steer_action, throttle_action, brake_action])
    
    action_dict['steer'].append(steer_action)
    action_dict['brake'].append(brake_action)
    action_dict['throttle'].append(throttle_action)
    
    action_dict['torque'].append(car.bicycle_model.torque_action2value)
    action_dict['steer'].append(car.bicycle_model.steer_action2value)
    
    action_dict['Ffx'].append(car.bicycle_model.Ffx)
    action_dict['Ffy'].append(car.bicycle_model.Ffy)
    action_dict['Frx'].append(car.bicycle_model.Frx)
    action_dict['Fry'].append(car.bicycle_model.Fry)
    
    

#dump trajectory
trajectory = np.array(car.bicycle_model.car_traj_arr).T
dump_f = {'car_x': trajectory[0], 'car_y': trajectory[1], 'car_phi': trajectory[2],
          'car_vx': trajectory[3], 'car_vy': trajectory[4]}
dump_f.update(action_dict)

# pickle.dump(dump_f, open('dynamics_test.pkl', 'wb'))
pickle.dump(dump_f, open('dynamics_test_torque_fix.pkl', 'wb'))


    