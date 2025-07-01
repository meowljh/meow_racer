'''ppo_algorithm.py
[Paper] Proximal Policy Optimization Algorithms
- On-Policy Method
- Simplified version of TRPO (Trust Region Policy Optimization)
- As all other policy optimization algorithms, it aims to maximize the J(s_t) objective function.
- Instead of the KL divergence constraint of the TRPO, PPO uses clipping to make the same effect of controlling the step size of the policy network update.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
from common.common_agent import CommonAgent
from common.common_buffer import RolloutBuffer
from ppo.ppo_network import PPO_MLP_Policy, PPO_MLP_Value

class PPO(CommonAgent):
    def __init__(self, 
                 env,
                 actor_hidden_dims,
                 critic_hidden_dims,
                 actor_optimizer_config:dict,
                 critic_optimizer_config:dict,
 
                 lr_decay:bool=True,
                 clip_epsilon:float=0.2,
                 batch_size:int=128,
                 discount_factor:float=0.99,
                 lamb_da:float=0.95,
                 num_epochs:int=10,
                 critic_loss_weight:float=1.0,
                 entropy_loss_weight:float=0.1,

                 hidden_activation_critic=nn.ReLU(),
                 hidden_activation_actor=nn.Tanh(),
                 
                 ):
        super(PPO, self).__init__(env=env)
        self.env = env
        self.optimizer_config = {
            'actor': actor_optimizer_config, 
            'critic': critic_optimizer_config
        }

        self.state_space = env.observation_space
        self.action_space = env.action_space
 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.actor_network = PPO_MLP_Policy(self.state_space, self.action_space, actor_hidden_dims, hidden_activation_actor)
        self.critic_network = PPO_MLP_Value(self.state_space, self.action_space, critic_hidden_dims, hidden_activation_critic)
        self.actor_network.to(self.device);self.critic_network.to(self.device)

        self.actor_optimizer = self._load_optimizer(actor_optimizer_config['name'], net=self.actor_network, lr=actor_optimizer_config['lr'], confs=actor_optimizer_config)
        self.critic_optimizer = self._load_optimizer(critic_optimizer_config['name'], net=self.critic_network, lr=critic_optimizer_config['lr'], confs=critic_optimizer_config)

        self.buffer = RolloutBuffer() #on-policy이기 때문에 예전의 policy로 얻은 trajectory 데이터를 재사용할 일이 없음. 따라서 그냥 REINFORCE와 동일하게 RolloutBuffer 사용#

        self.clip_epsilon = clip_epsilon
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.lamb_da = lamb_da
        self.num_epochs = num_epochs
        self.entropy_loss_weight = entropy_loss_weight
        self.critic_loss_weight = critic_loss_weight
        self.lr_decay = lr_decay

    @torch.no_grad()
    def act(self, state, training:bool=True):
        self.actor_network.train(mode=training)
        state = torch.as_tensor(state, dtype=torch.float, device=self.device)

        mu, std = self.actor_network(state)
        z = torch.normal(mean=mu, std=std) if training else mu
        sampled_action = torch.tanh(z)


        return {'action': sampled_action.detach().cpu().numpy()}

    # def process(self, transitions, step):
    def step(self, transitions, step):
        result = None
        self.buffer.store(transitions)

        if transitions[0]['done']:
            result = self.learn()
            if self.lr_decay:
                self.learning_rate_decay(step,
                                         optimizers=[
                                             self.actor_optimizer, self.critic_optimizer
                                         ])
                
        return result

    def learn(self):
        self.actor_network.train()
        self.critic_network.train()

        transitions = self.buffer.sample()
        state, action, reward, s_prime, done = transitions['state'], transitions['action'], transitions['reward'], transitions['state_prime'], transitions['done']
        s_t1, action, reward, s_t2, done = map(lambda x: torch.as_tensor(x, dtype=torch.float, device=self.device), [state, action, reward, s_prime, done])
        # r = r.unsqueeze(1);d = d.unsqueeze(1)
        

        ### (1) Calculate Generalzed Advantage and log probability of the action ###
        with torch.no_grad():
            reward_t = reward
            v_t_prime = self.discount_factor * (1-done) * self.critic_network(s_t2)
            v_t = self.critic_network(s_t1)

            delta_t = (reward_t + v_t_prime) - v_t ##advantage function of policy executed on timestep t##
            adv = torch.clone(delta_t)
            ret = torch.clone(reward_t)

            for t in range(len(ret)-2, -1, -1): ##just like dynamic programming##
                adv[t] += (1-done[t]) * self.discount_factor * self.lamb_da * adv[t+1]
                ret[t] += (1-done[t]) * self.discount_factor * ret[t+1]
            
            mu, std = self.actor_network(s_t1)
            gauss_dist = torch.distributions.Normal(loc=mu, scale=std)
            z = torch.atanh(torch.clamp(action, -1.+1e-7, 1.-1e-7))
            log_prob_old = gauss_dist.log_prob(z).sum(dim=-1, keepdims=True)

        adv = adv.view(-1, 1)
        ret = ret.view(-1, 1)
        

        ### (2) Train the actor network and critic network n_epocs ###
        data_set = torch.utils.data.TensorDataset(s_t1, action, ret, adv, log_prob_old)
        data_loader = torch.utils.data.DataLoader(data_set, batch_size=self.batch_size, shuffle=True)
        for e in range(self.num_epochs):
            actor_loss_arr, critic_loss_arr, entropy_bonus_arr = [], [], []
            for batch in data_loader:
                s_, a_, ret_, adv_, log_prob_old_ = batch

                value = self.critic_network(s_)
                critic_loss = F.mse_loss(value, ret_)

                mu, std = self.actor_network(s_)
                gauss_dist = torch.distributions.Normal(loc=mu, scale=std)
                z = torch.atanh(torch.clamp(a_, -1.+1e-7, 1.-1e-7))
                log_prob = gauss_dist.log_prob(z).sum(dim=-1, keepdims=True)

                ratio = (log_prob - log_prob_old_).exp()
                surrogate_loss_1 = adv_ * ratio
                surrogate_loss_2 = adv_ * torch.clamp(ratio,
                                                      1.-self.clip_epsilon,
                                                      1.+self.clip_epsilon)
                actor_loss = -torch.min(surrogate_loss_1, surrogate_loss_2).mean()
                entropy_bonus = -gauss_dist.entropy().mean()

                final_loss = actor_loss + self.critic_loss_weight*critic_loss + self.entropy_loss_weight*entropy_bonus
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                final_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                actor_loss_arr.append(actor_loss.item())
                critic_loss_arr.append(critic_loss.item())
                entropy_bonus_arr.append(entropy_bonus.item())

            
        result = {
            'actor_loss': np.mean(actor_loss_arr),
            'critic_loss': np.mean(critic_loss_arr),
            'entropy_bonus': np.mean(entropy_bonus_arr)
        }

        return result

                      
    def load(self, path):
        self.actor_network.load_state_dict(torch.load(f"{path}/actor.pth"))
        self.critic_network.load_state_dict(torch.load(f"{path}/critic.pth"))
        
        self.actor_optimizer.load_state_dict(torch.load(f"{path}/actor_optim.pth"))
        self.critic_optimizer.load_state_dict(torch.load(f"{path}/critic_optim.pth"))

    def save(self, path):
        torch.save(self.actor_network.state_dict(), f"{path}/actor.pth")
        torch.save(self.critic_network.state_dict(), f"{path}/critic.pth")

        torch.save(self.actor_optimizer.state_dict(), f"{path}/actor_optim.pth")
        torch.save(self.critic_network.state_dict(), f"{path}/critic_optim.pth")