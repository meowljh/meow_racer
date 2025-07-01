import torch
import torch.nn as nn

import os, sys
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
from common.common_agent import CommonAgent

class TRPO(CommonAgent):
    def __init__(self,
                 env):
        super(TRPO, self).__init__(env=env)
        