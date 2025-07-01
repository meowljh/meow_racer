import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle
from envs.utils import get_track_boundary

######################################################
ROOT = os.path.dirname(os.path.abspath(__file__))
EXP_ROOT = f"{ROOT}/experiments"
######################################################
def _track_bound():
    track_dict = pickle.load(open('statics/nam_c_track.pkl', 'rb'))
    left, right = get_track_boundary(np.array(track_dict['x']), np.array(track_dict['y']), np.array(track_dict['phi']))
    return left, right

def draw_base_track(ax, ax_idx:int=None):
    left, right = _track_bound()
    if ax_idx is None:
        ax.scatter(left.T[0], left.T[1], s=1, c='k', alpha=0.2)
        ax.scatter(right.T[0], right.T[1], s=1, c='k', alpha=0.2)
    else:
        ax[ax_idx].scatter(left.T[0], left.T[1], s=1, c='k', alpha=0.2)
        ax[ax_idx].scatter(right.T[0], right.T[1], s=1, c='k', alpha=0.2)
        


if __name__ == "__main__":
    EXP_NAME = ''
    