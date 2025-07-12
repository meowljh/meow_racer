import numpy as np
import yaml
from scipy.interpolate import CubicSpline, interp1d
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
import pickle
from vehicle.vehicle import RaceCar
from vehicle.vehicle_dynamics import RaceCar_Dynamics
from vehicle.vehicle_model import vehicleModel
from track.generate_random import Nam_TrackGenerator
from track.generate_circle import Circle_TrackGenerator
from reward.vehicle_status_check import OffCourse_Checker
from pid_controller import Steer_PID_Controller, Torque_PID_Controller
from simulator.main.pygame_sim import _render_all_pygame

class OffCourse_Checker_PID(OffCourse_Checker):
    def __init__(self):
        super().__init__(car_obj=None)
    
    def _reset(self, track_dict, bicycle_model, vehicle_model):
        self.track_dict = track_dict
        self.bicycle_model = bicycle_model
        self.vehicle_model = vehicle_model
        
        track_theta = np.array(track_dict['theta'])
        track_centerX, track_centerY = np.array(track_dict['x']), np.array(track_dict['y']) #남양 트랙만 center point는 이미 처음과 끝이 매칭 되어 있음
        track_centerX[-1] = track_centerX[0];track_centerY[-1] = track_centerY[0]
        theta_center_spline = CubicSpline(track_theta, np.vstack([track_centerX, track_centerY]).T, bc_type='periodic')
        self.theta_center_spline = theta_center_spline
        
        self.track_half_width = 3.5
        self.off_course_time = 0
        self.is_off_course = False
        
#(0) simulation configurations
DT = 0.01
STEER_CONF_DICT = {"Kp": 0.2, "Ki": 0., "Kd": 0.0005}
TORQUE_CONF_DICT = {"vel_target": 12, "Kp": 1000, "Ki": 0., "Kd": 0.0001,}

TRACK_MODE = 'nam' # 'circle' #'oval' #

SIMULATE_CFG_PATH = f'{ROOT}/conf/simulate/default.yaml'


########################################################################################################
#(1) setup track
if TRACK_MODE == 'nam':
    nam_track_cfg_path = f'{ROOT}/statics/nam_c_track.pkl'
    nam_track_gen = Nam_TrackGenerator(
        track_width=7,
        nam_track_path=nam_track_cfg_path,
        min_num_ckpt=4,
        max_num_ckpt=12
    )
    nam_track_gen._generate()
    track_dict = nam_track_gen._calculate_track_dict()
elif TRACK_MODE == 'circle':
    circle_track_gen = Circle_TrackGenerator(
        track_width=7,
        min_num_ckpt=4, max_num_ckpt=16
    )
    circle_track_gen._generate(
        radius=100, 
        d_angle_deg=3,
    )
    track_dict = circle_track_gen._calculate_track_dict()
elif TRACK_MODE == 'oval':
    oval_track_gen = Circle_TrackGenerator(
        track_width=7,
        min_num_ckpt=4, max_num_ckpt=16
    )
    oval_track_gen._generate(
        radius=100, d_angle_deg=3, straight_length=100, d_straight=2, initial_phi_deg=0
    )
    track_dict = oval_track_gen._calculate_track_dict()
    
    
def generate_splines(track_dict):
    track_theta = np.array(track_dict['theta'])
    track_centerX, track_centerY = np.array(track_dict['x']), np.array(track_dict['y'])
    track_leftX, track_leftY = np.array(track_dict['left']).T[0], np.array(track_dict['left']).T[1]
    track_rightX, track_rightY = np.array(track_dict['right']).T[0], np.array(track_dict['right']).T[1]
    track_phi = np.array(track_dict['phi'])
    track_kappa = np.array(track_dict['kappa'])

    track_centerX[-1] = track_centerX[0];track_centerY[-1] = track_centerY[0]

    track_leftX[-1] = track_leftX[0];track_leftY[-1] = track_leftY[0]

    track_rightX[-1] = track_rightX[0];track_rightY[-1] = track_rightY[0]

    theta_center_spline = CubicSpline(track_theta, np.vstack([track_centerX, track_centerY]).T, bc_type='periodic')
    theta_left_spline = CubicSpline(track_theta, np.vstack([track_leftX, track_leftY]).T, bc_type='periodic')
    theta_right_spline = CubicSpline(track_theta, np.vstack([track_rightX, track_rightY]).T, bc_type='periodic')
    phi_spline = interp1d(track_theta, track_phi, kind='linear', fill_value='extrapolate')
    kappa_spline = interp1d(track_theta, track_kappa, kind='linear', fill_value='extrapolate')
    
    
    return theta_center_spline, theta_left_spline, theta_right_spline, phi_spline, kappa_spline

#(2) setup bicycle model
bm = RaceCar_Dynamics(dt=DT)
vehicle_config_path = f'{ROOT}/environment/vehicle/jw_config.yaml'
bm._reset(
    config_yaml_path=vehicle_config_path,
    init_x=track_dict['x'][0],
    init_y=track_dict['y'][0],
    init_phi=track_dict['phi'][0],
    init_kappa=track_dict['kappa'][0]
)
bm.Vx =  8 #15

car_obj = RaceCar(
    action_dim=3,
    dt=DT,
    world_dt=DT,
    aps_bps_weight=1,
    allow_both_feet=True,
    brake_on_pos_vel=True,
    normalize_aps_bps=False,
    schedule_brake_ratio=False, schedule_brake_ratio_scale=0, schedule_brake_episode=0,
    zero_force_neg_vel=True, always_pos_vel=True,
    allow_neg_torque=True,
    cfg_file_path=vehicle_config_path
)
car_obj.bicycle_model = bm



#(3) setup status checker
status_checker = OffCourse_Checker_PID()
status_checker._reset(
    track_dict=track_dict,
    bicycle_model=bm,
    vehicle_model=bm.vehicle_model
)

#(4) setup the PID controller
steer_obj = Steer_PID_Controller(**STEER_CONF_DICT)
torque_obj = Torque_PID_Controller(**TORQUE_CONF_DICT)

#(5) setup the splines
theta_center_spline, theta_left_spline, theta_right_spline, phi_spline, kappa_spline = generate_splines(track_dict)

#################################################### 
####################### MAIN #######################

TERMINATED = False
x_arr = np.array([bm.car_x, bm.car_y, bm.car_phi, bm.Vx, bm.Vy, bm.Omega])
t=0
torque_control_arr = []
steer_control_arr = []

screen, clock = None, None

while not TERMINATED:
    screen, clock = _render_all_pygame(
        render_cfg=yaml.load(open(SIMULATE_CFG_PATH), Loader=yaml.FullLoader),
        car_obj=car_obj,
        track_dict=track_dict,
        t=t,
        screen=screen,
        clock=clock,
        render_car_state=True
    )
    
    torque_control = torque_obj._control(bm)
    # torque_control = 100 # 200
    steer_control = steer_obj._control(bm)
    
    ###constraint on the control values###
    steer_control = np.clip(steer_control, -0.4, 0.4)
    torque_control = np.clip(torque_control, -18000.0, 3727.5)
    
    torque_control_arr.append(torque_control)
    steer_control_arr.append(steer_control)
    
    x_arr = bm.cartesian_dynamics_2(
        x=x_arr,
        u=np.array([torque_control, steer_control]),
        allow_neg_torque=True
    )
    
    bm.track_reference_error(
        track_dict,
        theta_center_spline, theta_left_spline, theta_right_spline, phi_spline, kappa_spline
    )
    bm._calculate_Ec_w_spline(
        theta_center_spline, theta_left_spline, theta_right_spline
    )
    bm._calculate_Ephi_w_spline(
        phi_spline
    )
    bm._calculate_discrete_error()
    
    track_dict = bm.update_track_tile_stats(track_dict)
    
    t += DT
    TERMINATED = status_checker._off_course_tire()
    TERMINATED = sum(track_dict['passed']) == len(track_dict['passed'])

# pygame.close()
save_dict = {
    'TORQUE_PID_CONF' : TORQUE_CONF_DICT,
    'STEER_PID_CONF': STEER_CONF_DICT,
    'BM': bm.__dict__,
    'TORQUE': torque_control_arr,
    'STEER': steer_control_arr
    # 'CONTROL': car_obj.action2control
}

pickle.dump(save_dict, open(f"{os.path.dirname(os.path.abspath(__file__))}/pid_run_nam.pkl", 'wb'))