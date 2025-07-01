from stable.common.buffers import ReplayBuffer
from stable.common.type_aliases import (
    ReplayBufferSamples
)
from typing import Union
import numpy as np
import torch as th
from gymnasium import spaces

class NStepReplayBuffer(ReplayBuffer):
    """
    NStep Replay buffer used in off-policy algorithms like SAC/TD3

    :param n_steps: Number of steps for TD calculation
    
    ---------------------------------------------------------------
    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param handle_timeout_termination: Handle timeout termination
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
                 ):
        super().__init__(buffer_size=buffer_size, observation_space=observation_space, action_space=action_space,
                         device=device, n_envs=n_envs)
        
        np.random.seed(42)
        
        self.n_steps = n_steps
        self.gamma = gamma
        # breakpoint()
        ## add additional dimension to save the episode_id information
        ### should we add the additional dimension only to the observation object??
        ### NO!! 우선은 추가적인 dimension을 여기에 추가하기보다는 그냥 별도의 episode id 정보를 저장하는 array를 추가하는게 맞음!!
        ## 현재 observation에 대해서는 당연히 추가로 넣어 주어야 함.
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
        
        # self.episode_count = [0 for _ in range(self.n_envs)]
        self.episodes = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32) ##최대가 2^32-1
        
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
            # if sum(is_done) == 1:
            #     breakpoint()
            if self.handle_timeout_termination:
                self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

            self.pos += 1
            if self.pos == self.buffer_size:
                self.full = True
                self.pos = 0
    
    def _calc_n_step_reward(self, env, batch_inds, env_idxs):
        B = len(batch_inds)
        rewards = np.zeros(B).astype(np.float32)
        ##끝났는지의 여부에 해대서 False로 우선 설정
        dones = [False for _ in range(B)]
        
        new_batch_inds = [batch_inds[i] for i in range(B)]
        
        gamma_arr = [self.gamma ** self.n_steps for _ in range(B)]
        
        for b in range(B):
            for n in range(self.n_steps): ## 0~n_step-1
                ii = (batch_inds[b] + n) % self.buffer_size ## 단순히 batch_inds[b] + n으로 하는데 index error이 발생하지 않음.. 왜지??
                dones[b] = self.dones[ii, env_idxs[b]] * (1-self.timeouts[ii, env_idxs[b]])
                # dones[b] *= self.dones[ii, env_idxs[b]] * (1-self.timeouts[ii, env_idxs[b]]) ## timeout 되는 경우는 제외하고 episode에 대해서 done 처리 하기 ### -> 0인데 1을 곱해봤자 계속 0이 될 것임. dones 배열이 업데이트가 안되는 모양새였음.
                ## normalizing은 언제 하는게 좋을지..???? gamma로 가중치 부여하기 전에???
                ## 근데 reward의 running mean, std로 normalize를 해야 하기 때문에 각각의 reward에 대해서 normalize를 해주는 것이 맞을 것 같음. 아닌가???
                rewards[b] += (self.gamma ** n) * self._normalize_reward(reward = self.rewards[ii, env_idxs[b]], env=env)
                # if self.dones[batch_inds[b] + n, env_idxs[b]] == True:
 
                if self.dones[ii, env_idxs[b]] == True: ## episode가 끝났으면 여기까지만 처리 해주기 
                    # new_batch_inds[b] = n + batch_inds[b]
                    new_batch_inds[b] = ii
                    gamma_arr[b] = self.gamma ** (n+1)
                    break
                
            '''[0304] possible bug fix for N-Step!
            - dones처리 할 때 self.dones를 업데이트 해 버렸다. 실제로는 그렇게 하는게 아니라 _get_samples() 함수에 dones 배열을 반환해 주어야 하는 것이었음.
            - 그러나 결과적으로는 현재 reward function이 "현재 time step에서 추가로 밟은 tile의 개수가 많을수록" reward가 늘어나기 떄문에 n-step TD를 해주는 경우에
            계속 "현재 action을 빨리 밟아서 다음 step n개에 대해서만큼은" 빠른 속도를 유지하도록 한다. 
            게다가 critic의 학습 수렴도 빠르지 않기 때문에 target-Q value가 잘못 예측이 되고, 동시에 critic의 value function을 maximize하는 policy를 학습하게 되는 actor은 잘못된 policy를 학습하게 되는 것이다.
            
            -> 결국에 reward 값이 커지는데 기여를 하는 값으로 다른 가치를 추가해 주는 것도 방법이다.'''
            # if n == self.n_steps-1:
            #     dones[b] = self.dones[(batch_inds[b]+self.n_steps) % self.buffer_size, env_idxs[b]] * (1 - self.timeouts[(batch_inds[b] + self.n_steps)%self.buffer_size, env_idxs[b]])
                
        dones = np.array(dones).astype(np.float32)
        new_batch_inds = np.array(new_batch_inds)
        self.gamma_arr = np.array(gamma_arr).reshape(-1, 1)
        
        return rewards.reshape(-1, 1), dones.reshape(-1, 1), np.array(new_batch_inds)
            
    # def _get_samples(self, batch_inds:np.ndarray, env_idxs, env=None) -> ReplayBufferSamples:
    def _get_samples(self, batch_inds:np.ndarray, env=None) -> ReplayBufferSamples:
        
        '''[https://ymd_h.gitlab.io/cpprb/features/nstep/]
        혹시 몰라서 공식 문서 내용도 참고해 봤는데, 동일하게 구현하였음을 확인하였다.
        - next items = N step 후의 observation s_(t+N)
        - reward = r_t + gamma * r_(t+1) + gamma^2 + r_(t+2) + .. + gamma^(N-1) * t_(t+N-1)
        - done = 1 - ((1-d_t) * (1-d_(t+1)) *.. *(1-d_(t+N-1)))
        '''
        
        env_idxs = np.random.randint(0, high=self.n_envs, size=(len(batch_inds)))
        
        obs = self._normalize_obs(self.observations[batch_inds, env_idxs, :], env)
        ## critic에 넣어주는 행동은 현재 time step에 해당하는 action으로 그대로 사용.
        # critic의 예측과의 비교대상이 되는 target Q-value를 n-step 후의 cummulative reward로 대신하는 것이다.
        action = self.actions[batch_inds, env_idxs, :] ## critic에 입력으로 넣어줄 것임. 때문에 Q(s_t, a_t)와 target_value = r_t + gamma * r_(t+1) + .. + Q_target(s_(t+n), a_(t+n)') * dones * (gamma ^ n)의 TD loss를 통해서 critic을 업데이트 하면 된다.
        ## target critic은 가중치 기반으로 soft update된다.
        # action = self.actions[batch_inds + self.n_steps, env_idxs, :]
        ## dones처리 하는 부분에서 nstep 후의 terminate 여부를 buffer에서 sampling 했어야 함
        # dones = (self.dones[batch_inds+self.n_steps, env_idxs] * (1- self.timeouts[batch_inds+self.n_steps, env_idxs])).reshape(-1,1)
        rewards, dones, new_batch_inds = self._calc_n_step_reward(env, batch_inds, env_idxs)
        ##### _calc_n_step_reward에서 s_(t+n-1)까지의 누적 reward의 합을 계산하고 new_batch_inds도 마찬가지일 것이다.
        ##### 따라서 우리는 여기서도 1-step TD와 마찬가지로 newt_observations에 대해서 new_batch_inds로 인덱싱을 하여
        ##### target critic이 상태 s_(t+n)과 그 상태에 대한 현재 actor의 action a'_(t+n)에 대한 행동 가치 함수의 값을 뽑아내도록 한다.
        ##### 그러면 critic은 (gradient update가 되는)이 (s_t, a_t)에 대한 행동 가치 함수를 뽑아내어 이 두 값의 temporal difference로 학습이 된다.
        if not self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.next_observations[new_batch_inds, env_idxs, :], env) 
        else:
            next_obs = self._normalize_obs(self.observations[(new_batch_inds+1)%self.buffer_size, env_idxs, :], env)
        
        
        # rewards = self._normalize_reward(rewards, env)
        # breakpoint()
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
        # batch_inds = []
        # env_idxs=  []
        
        # for bs in range(batch_size):
        #     env_indice  = np.random.randint(0, high=self.n_envs, size=1)
        #     idx = np.random.randint(lower_bound, upper_bound)
        #     # while True:
        #     #     idx= np.random.randint(lower_bound, upper_bound)
        #     #     if self.episodes[idx, env_indice] == self.episodes[idx+self.n_steps, env_indice]:
        #     #         break
        #     batch_inds.append(int(idx))
        #     env_idxs.append(int(env_indice))
        #     # env_idxs.append(env_indice)
        '''[0305] possible randomness fix
        - batch size 전체에 대해서 iteration을 돌게 될에 비해서 전체 batch size 크기만큼의 random indices를 추출하는 것이 훨씬 random 측면에서 효과적일듯? 아닐수도
        '''
        if self.optimize_memory_usage:
            if self.full:
                batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
            else:
                batch_inds = np.random.randint(0, self.pos, size=batch_size)
        else:
            batch_inds = np.random.randint(lower_bound, upper_bound, size=batch_size)
        # batch_inds, env_idxs = np.array(batch_inds), np.array(env_idxs)
        
        # return self._get_samples(batch_inds, env_idxs, env=env)
        return self._get_samples(batch_inds, env=env)

    
                
                
        
        
        
        