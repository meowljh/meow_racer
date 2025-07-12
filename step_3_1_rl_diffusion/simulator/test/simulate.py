import numpy as np
'''make simple PID controller to go along the center line?'''

def pid_controller():
    pass


def run_simulation(track_generator, car_obj, max_run_step, terminate_Ec_thresh:float):
    '''
    @track_generator: track generator to generate tracks whenever the episode terminates
    @car_obj: the object that should be visualized on the simulation screen
    @max_run_step: maximum number of steps to run simulation
    @terminate_Ec_thresh: the threshold value for E_c of the car to consider termination validity
    '''
    step_cnt = 0
    while (step_cnt < max_run_step):
        track_generator._generate()
        track_dict = track_generator._calculate_track_dict()
        car_obj._reset(track_dict = track_dict)
        
        step_cnt += 1
        
        