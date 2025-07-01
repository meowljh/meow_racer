'''reinforce_algorithm.py
'''
import os, sys
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)

import torch
from torch.distributions import Normal
import numpy as np

from common.common_agent import CommonAgent
from common.common_network import MLPGaussianPolicy
from common.common_buffer import RolloutBuffer

class Reinforce(CommonAgent):
    def __init__(self, 
                 env, 
                 hidden_dims,
                 run_step:int,
                 optimizer_config:dict,
                 activation:str='relu',
                 action_activation:str='tanh',
                 use_standardization:bool=False,
                 gamma:float=0.99, #discount factor 
                 lr_decay:bool=True
                 ):
        super(Reinforce, self).__init__()
        self.env = env
        self.gamma = gamma
        self.run_step = run_step 
        self.lr_decay = lr_decay
        self.hidden_dims = hidden_dims
        self.optimizer_config = optimizer_config
        self.use_standardization = use_standardization
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
         
        self.policy_network = MLPGaussianPolicy(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=hidden_dims,
            activation_fn=self._load_activation_fn(activation),
            action_activation_fn=self._load_activation(action_activation),
        )
        
        self.policy_optimizer = self._load_optimizer(
            optmizer_name=self.optimizer_config['policy']['name'],
            net=self.policy_network,
            lr=self.optimizer_config['policy']['lr'],
            confs=self.optimizer_config['policy']
        )
        
        self.buffer = RolloutBuffer()
        
    def learn(self):
        self.policy_network.train()
        transitions = self.buffer.sample()
        state, action, reward = transitions['state'], transitions['action'], transitions['reward'] #이 형태의 dict는 training 단계에서 만들어서 넣어줌#
        s, a, r = map(lambda x: torch.as_tensor(x, dtype=torch.float, device=self.device), [state, action, reward])
        r = r.unsqueeze(1)
        
        return_arr = torch.clone(r) #clone reward#
        for t in range(len(return_arr)-1, -1, -1):
            return_arr[t] += self.gamma * return_arr[t+1]
        
        if self.use_standardization:
            return_arr = (return_arr - return_arr.mean()) / (return_arr.std() + 1e-7)
            
        mu, std = self.policy_network(s)
        m = Normal(mu, std)
        z = torch.atanh(torch.clamp(
            a, -1.+1e-7, +1.-1e-7
        ))
        log_probs = m.log_prob(z)
        # 로그 안에서의 곱셈은 로그들의 덧셈이기 떄문에 .sum() 사용 #
        log_probs = log_probs.sum(dim=-1, keepdim=True) #다차원의 행동 공간을 갖기 때문에 각 차원마다의 행동의 log probability를 얻으면, 이를 곱해주어야 최종 행동의 확률을 계산할 수 있다. (물론 각 행동들을 독립적이라고 가정하기 때문이다.) 
        
        
        policy_value_func = (return_arr * log_probs) # J(theta) #
        policy_loss = -policy_value_func.mean() # 어차피 gradient descent 할 때 minimize하도록 하니까 -를 붙여서 loss function minimize하는 것 처럼 #
        
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return {'policy_loss': policy_loss.item()}

    def process(self, transitions, step):
        result=None
        # process per step (하나의 episode를 완료 할 때까지 trajectory를 buffer에 쌓음) 
        # 하나의 trajectory에 들어가는 값은 (s_t, a_t, r_t) #
        self.buffer.store(transitions)
        
        # process per each episode (하나의 episode를 완료하였다면 그때 얻은 trajectory 사용하여 policy gradient update) #
        if transitions[0]['done']:
            result = self.learn()
            if self.lr_decay:
                self.learning_rate_decay(step,
                                         optimizers=[
                                             self.policy_optimizer]
                                         )  #episode가 terminate되거나 truncated 된 경우에 얻은 trajectory로 gradient update #
 
        return result
    
    
        
    @torch.no_grad()
    def act(self, state, training=True):
        self.policy_network.train(training)
        
        state = torch.as_tensor(state, dtype=torch.float, device=self.device)
        
        mu, std = self.policy_network(state)
        z = torch.normal(mu, std) if training else mu
        a = torch.tanh(z)
        
        return {'action': a.cpu().numpy().detach()}
        
        
        