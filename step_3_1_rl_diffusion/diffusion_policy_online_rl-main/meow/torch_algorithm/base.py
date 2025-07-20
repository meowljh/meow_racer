import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from pathlib import Path
import pickle

import numpy as np
import torch
import torch.nn as nn

from torch_utils.experience import Experience
# from torch_utils.persistence import make_persist
# from torch_utils.typing_utils import Metric

class Algorithm:
    def __init__(self):
        super().__init__()
    
    def update(self, data: Experience):
        info = self.stateless_update(data=data)
        return {k: float(v) for k, v in info.items() if not k.startswith('hist')}, {k: v for k, v in info.items() if k.startswith('hist')}
    
    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """off policy algorithm 부분에서 action 받아오기 위해서 agent에 입력으로 observation을 넣기전에 tensor 변환 + gpu device할당을 해 주어야 함."""
        action = self.agent.get_action(obs=obs)
        return action
    
    def get_deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        action = self.agent.get_deterministic_action(obs=obs)
        return action
    
    def warmup(self, data: Experience) -> None:
        obs = data.obs[0]
        self.update(data=data)
        self.get_action(obs=torch.tensor(obs, dtype=torch.float32, device=self.device))
        self.get_deterministic_action(obs=torch.tensor(obs, dtype=torch.float32, device=self.device))

    def save_policy(self, path) -> None:
        policy_state_dict = self.agent.policy.state_dict()
        with open(path, 'wb') as w:
            pickle.dump(policy_state_dict, w)
        w.close()

    def save(self, path) -> None:
        state_dicts = self.get_state_dicts()
        with open(path, 'wb') as w:
            pickle.dump(state_dicts, w)
        w.close()