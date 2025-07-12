from .generate_random import (Base_TrackGenerator, 
                Nam_TrackGenerator, Random_TrackGenerator,
)
from .generate_bezier import Bezier_TrackGenerator

import os, sys
sys.path.append("..")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
gen_args = {
    'min_num_ckpt': 4,
    'max_num_ckpt': 16,
    'track_radius': 200,
    'scale_rate': 1.,
    'min_track_turn_rate': 0.31,
    'track_turn_rate': 0.31,
    'track_detail_step': 5,
    'track_detail_step_check': 21
}
dummy_nam_gen = Nam_TrackGenerator(track_width = 7, nam_track_path = f"{ROOT}/statics/nam_c_track.pkl", **gen_args)

dummy_rand_gen = Random_TrackGenerator(track_width=7, **gen_args)

dummy_bezier_gen = Bezier_TrackGenerator(
    min_kappa=0.04, max_kappa=0.1,
    track_width=7.,
    **gen_args
)

__all__ = [Base_TrackGenerator, Nam_TrackGenerator, Random_TrackGenerator, Bezier_TrackGenerator]