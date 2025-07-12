from abc import ABC
import os, sys
import pandas as pd
import math
import numpy as np
from scipy.interpolate import interp1d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

"""
ref: https://ade-bitb.hmckmc.co.kr/projects/ADCD/repos/vti_poc/browse/mpcc/controller_side/vti_poc_nam_c_code/vti.h
"""


class APS_Mapping(ABC):
    def __init__(self):
        super().__init__()
        self.dVxBorder0 = 0.0 #m/s
        
        self.dAffSec1_p1 = 2.0533e-15
        self.dAffSec1_p2 = 3.7275e+03
        self.dVxBorder1 = 15.9646 #m/s
        
        self.dExpSec_p1 = 6.7039e+03
        self.dExpSec_p2 = -3.7643e-02
        self.dVxBorder2 = 52.9734 #m/s
        
        self.dAffSec2_p1 =  -716.9334
        self.dAffSec2_p2 = 3.9019e+04
        self.dVxBorder3 = 54.4248 #m/s
        
    def _get_APS_Torque(self, throttle_action, max_tau, car_vx):
        assert 0 <= throttle_action <= 1
        '''최대로 APS를 밟았을 때 APS torque값을 lateral vel에 따라 계산하면, 여기에 throttle action을 곱해야 함.'''
        if (car_vx < self.dVxBorder0):
            aps_torque = self.dAffSec1_p1 * car_vx + self.dAffSec1_p2
        elif (self.dVxBorder0 <= car_vx < self.dVxBorder1): #속도가 음수일때사 dVxBorder2 보다 작을때는 같게 처리함.
            aps_torque = self.dAffSec1_p1 * car_vx + self.dAffSec1_p2
        elif (self.dVxBorder1 <= car_vx < self.dVxBorder2):
            aps_torque = self.dExpSec_p1 * math.exp(car_vx * self.dExpSec_p2)
        elif (self.dVxBorder2 <= car_vx < self.dVxBorder3):
            aps_torque = self.dAffSec2_p1 * car_vx + self.dAffSec2_p2
        else:
            ##속도가 최대 속도를 넘어가면 accelerator을 밟아도 그로 인한 torque 값은 당연히 0이어야 함.
            ##물론 BPS 관련 매핑 이슈로 인해서 최대 속도를 넘어가는 경우가 별로 없었겠지만, BPS 에러 수정 후에는 이 부분을 몰랐다면 target speed reward가 아니었다면 문제가 분명히 되었을 것이다.
            # aps_torque = max_tau
            aps_torque = 0
        
        '''[0430] Error Fix: TyPo - did not apply the throttle action value to the APS Torque value.'''
        aps_torque *= throttle_action
        
        return aps_torque
        
#ref: C:\Users\7459985\Desktop\2024\24001_Visual_Track_Instructor\CE_제동맵.xlsx

class BPS_Mapping(ABC):
    def __init__(self):
        super().__init__()
        ref_pressure = np.array([
            0, 2, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70
        ], dtype=np.float32)

        ref_pressure /= max(ref_pressure) #normalized to ratio (0~1)
        self.ref_pressure = ref_pressure


        self.bps_map = {
            #mapping of KPH : Braking Force based on the pressure value
            #차속에 따라서 BPS 압력이 force로서 작용되는 비율이 달라짐
            #단위 bar -> N으로 바꾸기 위해서 100곱해야함.
            3 : np.array([0, 0, 1.5, 2.2, 3.5, 7, 10, 12, 18, 27, 38, 50, 65, 80, 90, 90, 90], dtype=np.float32),
            10: np.array([0, 0, 1.5, 2.2, 3.5, 7, 10, 12, 18, 27, 38, 50, 65, 90, 130, 160, 180], dtype=np.float32),
            20: np.array([0, 0, 1.1, 2.2, 3.5, 7, 12, 18, 25, 33, 42, 55, 74, 101, 130, 160, 180], dtype=np.float32),
            30: np.array([0, 0, 1.1, 2.2, 3.5, 7, 12, 18, 25, 33, 42, 55, 74, 101, 130, 160, 180], dtype=np.float32),
            50: np.array([0, 0, 1.1, 2.2, 3.5, 7.5, 12.5, 18.5, 26, 33.5, 43.5, 57.5, 76.5, 101, 130, 160, 180], dtype=np.float32),
            90: np.array([0, 0, 1.1, 2.2, 3.5, 7.5, 12.5, 18.5, 26, 33.5, 43.5, 57.5, 76.5, 101, 130, 160, 180], dtype=np.float32),
        } 
        
        #90kph가 딱 25m/s인데, 혹시 이 때문에 최대 차량의 속도가 계속 25m/s에 머물렀던 것 같음. 
        #그 이후의 가능한 속도 값들에 대해서도 추가를 해 주어여 할 것 같음.
        
        self.interp_dict = {key: interp1d(self.ref_pressure, value, kind='linear') for \
            key, value in self.bps_map.items()}

    def _get_BPS_Torque_Continuous(self, brake_action, min_tau, car_vx):
        assert 0 <= brake_action <= 1
        kph_vel_arr = np.array(list(self.bps_map.keys()))
        kph_car_vx = car_vx * 3.6 #ms -> kph
        
        bps_torque = abs(min_tau) #절댓값으로 사용해야 함.
        
        max_speed_kph = 200 #200kph == 55.5ms
        
        for i, vel in enumerate(kph_vel_arr):
            interp = self.interp_dict[vel]
            if i == 0:
                if kph_car_vx < kph_vel_arr[i]: #0~3 사이
                    bps_torque_a = 0
                    bps_torque_b = interp(brake_action) * 100
                    ratio = kph_car_vx / (kph_vel_arr[i] - 0)
                    bps_torque = (bps_torque_b - bps_torque_a) * ratio + bps_torque_a
                    return bps_torque
            else:
                if kph_vel_arr[i-1] <= kph_car_vx < kph_vel_arr[i]:
                    bps_torque_a = self.interp_dict[kph_vel_arr[i-1]](brake_action) * 100
                    bps_torque_b = interp(brake_action) * 100
                    ratio = (kph_car_vx - kph_vel_arr[i-1]) / (kph_vel_arr[i] - kph_vel_arr[i-1])
                    bps_torque = (bps_torque_b - bps_torque_a) * ratio + bps_torque_a
                    return bps_torque
        #returns min_tau value when the longitude velocity is larger than the max velocity (200kph)
        #최대 속도보다 작고 90kph보다 큰 경우에도 역시나 90kph에 해당하는 BPS map을 적용한다.
        if kph_car_vx <= max_speed_kph:
            interp = self.interp_dict[kph_vel_arr[-1]]
            bps_torque = interp(brake_action) * 100
            
            return bps_torque
    
        return bps_torque
    
    def _get_BPS_Torque(self, brake_action, min_tau, car_vx):
        kph_vel_arr = np.array(list(self.bps_map.keys()))
        kph_car_vx = car_vx * 3.6 #ms -> kph
        
        bps_torque = min_tau
        
        max_speed_kph = 200 #200kph == 55.5ms
        
        for i, vel in enumerate(kph_vel_arr):
            interp = self.interp_dict[vel]
            if i == 0:
                if kph_car_vx < kph_vel_arr[i]:
                    bps_torque = interp(brake_action) * 100 
                    return bps_torque
            else:
                if kph_vel_arr[i-1] <= kph_car_vx < kph_vel_arr[i]:
                    bps_torque = interp(brake_action) * 100
                    return bps_torque
        #returns min_tau value when the longitude velocity is larger than the max velocity (200kph)
        #최대 속도보다 작고 90kph보다 큰 경우에도 역시나 90kph에 해당하는 BPS map을 적용한다.
        if kph_car_vx <= max_speed_kph:
            interp = self.interp_dict[kph_vel_arr[-1]]
            bps_torque = interp(brake_action) * 100
            return bps_torque
        
        return bps_torque