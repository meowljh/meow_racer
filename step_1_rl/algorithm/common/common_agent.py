import torch
import torch.nn.functional as F

import numpy as np
import os

from abc import *

class CommonAgent(ABC):
    def __init__(self, env):
        super(CommonAgent, self).__init__()
        self.env = env
    
    @abstractmethod
    def learn(self):
        return {
            "loss": None,
        }
    
    @abstractmethod
    # def process(self, transitions, step):
    def step(self, transitions, step):
        return {}
    
    @abstractmethod
    def act(self, state):
        action = None
        action_dict = {"action": action}
        return action_dict
    
    @abstractmethod
    def save(self, path):
        pass
    
    @abstractmethod
    def load(self, path):
        pass
    
    def sync_in(self, weights):
        '''내부 네트워크 이름이 network로 정의되지 않은 경우에는 내부적으로 function을 새롭게 짜야 한다.'''
        self.network.load(weights)
    
    def sync_out(self, device:str='cpu'):
        weights = self.network.state_dict()
        for k, v in weights.items():
            weights[k] = v.to(device)
        sync_item = {
            'weights': weights
        }
        return sync_item
        
    def _load_activation(self, activation_name):
        if activation_name.lower() == 'relu':
            return F.relu
        elif activation_name.lower() == 'tanh':
            return F.tanh
        elif activation_name.lower() == 'sigmoid':
            return F.sigmoid
        elif activation_name.lower() == 'gelu':
            return F.gelu
    
    def _load_optimizer(self, optimizer_name, net, lr, confs):
        if optimizer_name.lower() == 'adam':
            return torch.optim.Adam(
                params=net.parameters(),            
                lr=lr,
                betas=confs['betas'], #first momentum, second momentum 각각에 대한 weighted average를 위한 weight 값#
                eps=confs['eps'], #별 의미 없고, parameter 업데이트 할 때 second momentum이 0일것을 대비해 division error 막음#
                weight_decay=confs['weight_decay']
            )
        elif optimizer_name.lower() == 'sgd':
            return torch.optim.SGD(
                params=net.parameters(),
                lr=lr,
                momentum=confs['momentum'],
                weight_decay=confs['weight_decay'], #for L2 penalty loss#
                nesterov=(confs['momentum'] != 0), #momentum factor used when using nesterov optimization#
            )
    
    def learning_rate_decay(self, step, optimizers=None, mode:str='cosine'):
        if mode == 'cosine':
            weight = np.cos((np.pi / 2) * (step / self.num_epochs))
        elif mode == 'linear':
            weight = 1 - (step / self.num_epochs)
        elif mode == 'sqrt':
            weight = (1 - (step / self.num_epochs)) ** 0.5
        else:
            raise Exception(f"invalid learning rate decay mode : {mode}")
        
        if optimizers is None:
            optimizers = [self.optimizer]
            
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        
        for optim in optimizers:
            for g in optim.param_groups:
                g["lr"] = optim.defaults["lr"] * weight
                
                