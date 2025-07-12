from .vehicle_dynamics import RaceCar_Dynamics
from .vehicle import RaceCar


import os, sys
import numpy as np
sys.path.append("..")
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# dummy_racecar = RaceCar(
#     action_dim=2,
#     dt=0.2 , #1/60,
#     aps_bps_weight=1.,
#     allow_both_feet=True,
#     cfg_file_path = f"{ROOT}/vehicle/jw_config.yaml"
# )

