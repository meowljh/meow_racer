import math
import numpy as np
from abc import ABC

class PID_Controller(ABC):
    def __init__(self, Kp, Ki, Kd):
        super().__init__()
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        
class Steer_PID_Controller(PID_Controller):
    def __init__(self, Kp, Ki, Kd):
        super().__init__(Kp=Kp, Ki=Ki, Kd=Kd)
        self.integ = 0
        self.deriv = 0
        self.error = 0
        self.t = 0
        #### target for control aims the E_c and E_phi to be zero ####
        self.E_c_target = 0
        self.E_phi_target = 0
        
    def _control(self, bm):
        '''**현재 값 - 목표값**'''
        new_error = bm.E_c - self.E_c_target #self.E_c_target - bm.E_c
        self.integ += new_error
        self.deriv = (new_error - self.error) / bm.dt
        self.error = new_error
        
        control_val = sum([self.error * self.Kp,
                                self.integ * self.Ki,
                                self.deriv * self.Kd]) 
        
        self.t += bm.dt
        
        return control_val
        
        
class Torque_PID_Controller(PID_Controller):
    def __init__(self, 
                 vel_target,
                 Kp, Ki, Kd):
        super().__init__(Kp=Kp, Ki=Ki, Kd=Kd)
        self.integ = 0
        self.deriv = 0
        self.error = 0
        self.t = 0
        #### control for Torque value aims to match the car velocity to the target ####
        self.vel_target = vel_target
    
    def _control(self, bm):
        new_error = self.vel_target - bm.Vx #bm.Vx - self.vel_target
        self.integ += new_error
        self.deriv = (new_error - self.error) / bm.dt
        self.error = new_error
        
        control_val = sum([
            self.error * self.Kp,
            self.integ * self.Ki,
            self.deriv * self.Kd
        ])
        
        self.t += bm.dt
        
        return control_val
        
        
        
        
        
        