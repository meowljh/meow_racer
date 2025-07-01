from stable.common.buffers import ReplayBuffer
from stable.common.type_aliases import (
    ReplayBufferSamples
)
from typing import Union
import numpy as np
import torch as th
from gymnasium import spaces


class LambdaStepReplayBuffer(ReplayBuffer):
    """
    Lambda Step Replay buffer
    - 유의해야 하는 것은, TD(lambda) 방법은 주로 on-policy 정책에서 사용이 된다는 것이다.
        따라서 off policy인 SAC 알고리즘에서의 학습에는 적극적으로 사용되지 않을수도 있다.
        - 기본적으로 TD learning이라는 것은 On Policy를 전제로 두고 있기 때문에 SAC에서 TD(n)을 사용한다는 것이 말이 안되는 것일수도 있었다.
    - resolves the difficulty of selecting the n-step value
    - lambda-return considers all the n-step returns to update the state value, which can jointly take advantage of the TD and MC method.
    1. Forward view: MC와 동일하게 n-step까지 도달한 후 지금까지 얻어낸 return들을 가중 평균하여 lambda-return을 구함 (offline update)
        -> 0 < lambda < 1의 범위에 들어 있는 가중치로 n번째까지의 n-step TD 값들의 가중 평균을 계산해서 모든 step에 대한 TD error로 업데이트를 한다.
    2. Backward view: 각 state의 eligibility trace값을 사용해서 각 step마다 state value를 실시간으로 업데이트 (online update)
    """
    def __init__(self, 
                 buffer_size: int,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 device: Union[th.device, str]="auto",
                 n_envs: int=1,
                 optimize_memory_usage: bool=False,
                 handle_timeout_termination: bool=True,
                 gamma: float=0.99,
                 n_steps: int=3,
                 
                 step_lambda: float=0.99,
                 mode: str = "forward"
                 ):
        super().__init__(buffer_size=buffer_size, observation_space=observation_space, action_space=action_space,
                         device=device, n_envs=n_envs)
        self.step_lambda = step_lambda
        self.mode = mode
        
        np.random.seed(42)
        
        self.n_steps = n_steps
        self.gamma = gamma 
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)
        
        if not optimize_memory_usage:
            self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)
        
        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim),
            dtype=self._maybe_cast_dtype(action_space.dtype)
        )
        
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
         
        self.episodes = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)  
        
    def add(self, 
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos
            ) -> None:
        
            if isinstance(self.observation_space, spaces.Discrete):
                obs = obs.reshape((self.n_envs, *self.obs_shape))
                next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))
            
            # reshape action to handle multi-dim and discrete action spaces
            action = action.reshape((self.n_envs, self.action_dim))
            self.observations[self.pos] = np.array(obs)
            
            if self.optimize_memory_usage: ## next_observations 배열을 따로 attribute로 저장하지 않음.
                self.observations[(self.pos+1) % self.buffer_size] = np.array(next_obs)
            else:
                self.next_observations[self.pos] = np.array(next_obs)
            
            
            self.actions[self.pos] = np.array(action)
            self.rewards[self.pos] = np.array(reward)
            self.dones[self.pos] = np.array(done)
            
            is_done = (np.array(done) == True).astype(int)
            self.episodes[(self.pos+1)%self.buffer_size] = self.episodes[self.pos] + is_done
 
            if self.handle_timeout_termination:
                self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

            self.pos += 1
            if self.pos == self.buffer_size:
                self.full = True
                self.pos = 0
    
    def _calc_lambda_forward_step_reward(self, env, batch_inds, env_idxs):
        '''원래 TD(lambda)의 forward step은 episode가 끝날때까지의 모든 n-step return을 고려해야 해서 실제로
        적용 단계에서는 forward TD(lambda)는 사용하지 않는다.
        -> 해당 구현체는 
        '''
        B = len(batch_inds)
        rewards = np.zeros(B).astype(np.float32) 
        dones = [False for _ in range(B)]
        
        new_batch_inds = [batch_inds[_] for _ in range(B)]
        
        gamma_arr = [self.gamma ** self.n_steps for _ in range(B)]
        
        for b in range(B):
            lambda_target_G = 0 
            norm_val = 0
            for n in range(self.n_steps): ## 0~n_step-1
                ii = (batch_inds[b] + n) % self.buffer_size  
                dones[b] = self.dones[batch_inds[b]+n, env_idxs[b]] * (1-self.timeouts[batch_inds[b]+n, env_idxs[b]]) 
                # rewards[b] += (self.gamma ** n) * self._normalize_reward(reward = self.rewards[ii, env_idxs[b]], env=env)
                lambda_target_G += (self.gamma ** n) * self._normalize_reward(reward = self.rewards[ii, env_idxs[b]], env=env)
                rewards[b] += (self.step_lambda ** n) * lambda_target_G
                norm_val += self.step_lambda ** n
                
                if self.dones[ii, env_idxs[b]] == True: ## episode가 끝났으면 여기까지만 처리 해주기  
                    new_batch_inds[b] = ii
                    gamma_arr[b] = self.gamma ** (n+1)
                    
                    break
                
            '''[0304] possible bug fix for N-Step!
            - dones처리 할 때 self.dones를 업데이트 해 버렸다. 실제로는 그렇게 하는게 아니라 _get_samples() 함수에 dones 배열을 반환해 주어야 하는 것이었음.'''
            # if n == self.n_steps-1:
            #     dones[b] = self.dones[(batch_inds[b]+self.n_steps) % self.buffer_size, env_idxs[b]] * (1 - self.timeouts[(batch_inds[b] + self.n_steps)%self.buffer_size, env_idxs[b]])
            gamma_arr[b] /= norm_val
            
        dones = np.array(dones).astype(np.float32)
        new_batch_inds = np.array(new_batch_inds)
        self.gamma_arr = np.array(gamma_arr).reshape(-1, 1)
        # rewards *= (1-self.step_lambda) ## 이걸 곱하는 이유는 가중치의 합 때문임. lambda^n의 합이 n->inf일 때 등비수열의 합에 의해서 1/(1-lambda)가 되어서 normalizing을 위해 1-lambda를 곱해준다.
        
        return rewards.reshape(-1, 1), dones.reshape(-1, 1), np.array(new_batch_inds)
    
    def _calc_lambda_backward_step_reward(self, env, batch_inds, env_idxs):
        pass
          
    def _get_samples(self, batch_inds:np.ndarray, env=None) -> ReplayBufferSamples:
        
        '''[https://ymd_h.gitlab.io/cpprb/features/nstep/]
        혹시 몰라서 공식 문서 내용도 참고해 봤는데, 동일하게 구현하였음을 확인하였다.
        - next items = N step 후의 observation s_(t+N)
        - reward = r_t + gamma * r_(t+1) + gamma^2 + r_(t+2) + .. + gamma^(N-1) * t_(t+N-1)
        - done = 1 - ((1-d_t) * (1-d_(t+1)) *.. *(1-d_(t+N-1)))
        '''
        
        env_idxs = np.random.randint(0, high=self.n_envs, size=(len(batch_inds)))
        
        obs = self._normalize_obs(self.observations[batch_inds, env_idxs, :], env) 
        action = self.actions[batch_inds, env_idxs, :] 
        
        if self.mode == "forward":
            rewards, dones, new_batch_inds = self._calc_lambda_forward_step_reward(env, batch_inds, env_idxs) 
        elif self.mode == "backward":
            rewards, dones, new_batch_inds = self._calc_lambda_backward_step_reward(env, batch_inds, env_idxs)
            
        if not self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.next_observations[new_batch_inds, env_idxs, :], env) 
        else:
            next_obs = self._normalize_obs(self.observations[(new_batch_inds+1)%self.buffer_size, env_idxs, :], env)
        
         
        data = (
            obs, ## s_t ~ D
            action, ## a_t  ~ D
            next_obs, ## s_(t+n) ~ D
            dones, ## is_done(s_(t+n-1)) -> 만약에 t+n-1 번째에서 done이 되었다면, 다음 step의 observation에 대한 Q-value estimation은 target value에 고려할 필요가 없음.
            rewards ## r_t + gamma * r_(t+1) + gamma^2 * r_(t+2) + .. + gamma^(n-1) * r_(t+n-1)
        ) 
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
            
            
    def sample(self, batch_size:int, env=None) -> ReplayBufferSamples:
        lower_bound = 0
        upper_bound = self.buffer_size if self.full else self.pos
 
        if self.optimize_memory_usage:
            if self.full:
                batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
            else:
                batch_inds = np.random.randint(0, self.pos, size=batch_size)
        else:
            batch_inds = np.random.randint(lower_bound, upper_bound, size=batch_size)
 
        return self._get_samples(batch_inds, env=env)