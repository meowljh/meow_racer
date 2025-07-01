import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

def dummy_learn(policy_net, state, env):
    action_min, action_max = env.action_space.low, env.action_space.high #np.ndarray#
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    state = torch.as_tensor(state, dtype=torch.float).to(device)
    policy_net.to(device)
    
    mu, log_std_exp = policy_net(state)
    
    ## (1) calculate action ##
    action = torch.normal(mu, log_std_exp)
    action = torch.tanh(action)
    action = action.cpu().detach().numpy()
    action = 0.5 * (action_max - action_min) * (action + 1) + action_min #scale to original action max & min#
    
    ## (2) calculate log-prob ##
    m = Normal(mu, log_std_exp) 
    z = torch.atanh(
        torch.clamp(torch.as_tensor(action, dtype=torch.float, device=device),
                    -1.+1e-7, +1.-1e-7) #tan^-1은 -1과 1부근에서 무한대로 발산을 하기 때문에 clamping을 해 주어야 함#
    ) #action의 범위를 맞춰 주었었기 때문에 다시 확률 분포에서의 확률 값을 계산하기 위해서 (-inf, +inf)의 범위로 바꿔 주어야 함#
    log_probs = m.log_prob(z).sum(dim=-1, keepdim=True)
    
    
    return action, log_probs
    
    
class MLPGaussianPolicy(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dims,
                 activation_fn=F.relu,
                 action_activation_fn=F.tanh):
        super(MLPGaussianPolicy, self).__init__()
        self.input_layer = nn.Linear(state_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.hidden_layers.append(
                nn.Linear(hidden_dims[i], hidden_dims[i+1])
            )
        
        self.mu_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)
        
        self.activation_fn = activation_fn
        self.action_activation_fn = action_activation_fn
    
    def forward(self, state):
        state = self.activation_fn(self.input_layer(state))
        for hidden in self.hidden_layers:
            state = self.activation_fn(hidden(state))
        
        mu = self.mu_layer(state)
        log_std = self.log_std_layer(state)
        '''policy network가 뱉은 행동이 특정 범위에 들어갈 수 있도록 해야 한다.
        일반적으로는 environment마다 행동의 최대/최소 범위가 다르지만, tanh를 사용해서 (-1, 1) 사이의 범위를 갖도록 한 뒤에
        행동의 범위를 "실제 각 action의 min/max로" 조절하는 것이 편하기는 하다.'''
        log_std = self.action_activation_fn(log_std)
        
        return mu, log_std.exp()
    
class MLPStateValue(nn.Module):
    def __init__(self,
                 state_dim, 
                 hidden_dims,
                 activation_fn=F.relu):
        super(MLPStateValue, self).__init__()
        self.input_layer = nn.Linear(state_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList([])
        
        for i in range(len(hidden_dims)-1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        self.activation_fn = activation_fn
    
    def forward(self, state):
        state = self.activation_fn(self.input_layer(state))
        for layer in self.hidden_layers:
            state = self.activation_fn(layer(state))
        state = self.output_layer(state)
        
        return state
        
if __name__ == "__main__":
    import gym
    env = gym.make("MountainCarContinuous-v0")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    hidden_dims = (128, 128)
    s, info = env.reset()
    
    net = MLPGaussianPolicy(
        state_dim, action_dim, hidden_dims
    )
    
    action, log_probs = dummy_learn(
        policy_net=net, state=s, env=env
    )
    