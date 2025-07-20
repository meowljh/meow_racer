import numpy as np
import torch
import random

def seed_all(seed:int):
    torch.manual_seed(seed=seed)
    np.random.seed(seed=seed)
    random.seed(seed)