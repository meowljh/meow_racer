import pickle
import numpy as np
import os, sys

"""
1. loads trained actor-critic
2. loads environment, vehicle based on the configuration of the experiment
3. tests the policy deterministically on the Nam-C racing track
"""
def _load_pkl_data(fpath):
    assert os.path.splittext(fpath) == '.pkl'
    pkl_data = pickle.load(open(fpath, 'rb'))
    return pkl_data


def _step_1_load_sac(saved_root):
    pass

def _step_2_load_experiment(conf_dict_path):
    conf_dict = _load_pkl_data(fpath=conf_dict_path)
    pass

def _step_3_test_nam_c_track():
    pass


