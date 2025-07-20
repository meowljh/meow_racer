from pathlib import Path

import pickle
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.dirname(ROOT))
sys.path.append(os.path.dirname(os.path.dirname(ROOT)))
# from environment.racedemia_env_v1 import Racedemia_Env

from typing import Callable, Optional, Tuple, Sequence

from gymnasium import Env #RacedemiaEnv is also inherited from the gymnasium Env
# from gymnasium.vector import VectorEnv #vectorized env는 당분간은 사용하지 않는 것으로..

import torch
import torch.nn as nn

from tensorboardX import SummaryWriter
from tensorboardX.summary import hparams

import numpy as np
from tqdm import tqdm

import wandb

from torch_algorithm.base import Algorithm
from torch_buffer.replay_buffer import ReplayBuffer
# from torch_env.vector import VectorEnv
from torch_utils.experience import Experience
from torch_trainer.accumulator import SampleLog, Interval, UpdateLog



class OffPolicyTrainer:
    def __init__(
            self,
            seed: int,
            device,
            env: Env,
            algorithm: Algorithm, #SAC / SDAC
            buffer: ReplayBuffer,
            log_path: Path,
            batch_size: int = 256, #parameter update시의 batch size
            start_step: int = 1000, #초기에 warmup (buffer 채우는 과정)을 몇번째 step까지 할 것인지
            total_step: int = int(1e6), #전체 train loop동안의 최대 step 수
            sample_per_iteration: int = 1, #한번의 iteration마다 몇번의 exploration 단계를 거치는지
            update_per_iteration: int = 1, #한번의 iteration마다 몇번의 backprob 단계를 거치는지
            #########################
            #train_env: Random Track#
            #evaluate_env: Nam Track#
            #########################
            evaluate_env: Optional[Env] = None,
            evaluate_every: int = 10000,
            evaluate_n_episode: int = 20, #연속적으로#
            sample_log_n_episode: int = 10,
            update_log_n_step: int = 1000,
            save_policy_every: int = 10000,
            save_value: bool = True,
            hparams: Optional[dict] = None,
            policy_pkl_template: str = "policy-{sample_step}-{update_step}.pkl",
            warmup_with: str = "random",  # "policy" or "random" #for the initialization of the replay buffer before training
    ):
        super().__init__()

        self.seed = seed

        self.device = device

        self.hparams = hparams

        """[TODO]
        add racedemia env static files to allow import of Racedemia_Env"""
        # self.is_racedemia = isinstance(env.unwrapped, Racedemia_Env)
        self.is_racedemia = False

        self.sl = SampleLog()
        self.ul = UpdateLog()

        self.sample_log_interval = Interval(sample_log_n_episode)
        self.save_policy_interval = Interval(save_policy_every)
        self.eval_policy_interval = Interval(evaluate_every)

        self.env = env
        self.algorithm = algorithm
        self.buffer = buffer

        self.is_vec = False

        self.log_path = log_path

        self.log_episode_dict = dict()

        #training hyperparameter setup#
        self.batch_size = batch_size
        self.start_step = start_step
        self.total_step = total_step
        self.sample_per_iteration = sample_per_iteration
        self.update_per_iteration = update_per_iteration
        self.warmup_with = warmup_with

        #evaluation hyperparameter setup#
        self.evaluate_env = evaluate_env
        self.evaluate_n_episode = evaluate_n_episode
        self.update_log_n_step = update_log_n_step

        #logging hyperparameter steup#
        self.save_policy_every = save_policy_every
        self.save_value = save_value
        self.policy_pkl_template = policy_pkl_template
        
        #init wandb
        wandb.init(
            project="racedemia_diffusion_online_rl",
            name=log_path.name,
            dir=log_path,
            group=env.spec.id
        )



    def evaluate_track(self): # -> tuple([Sequence[int], Sequence[float], bool]):
        """지정된 evaluate_n_episode만큼 연속적으로 트랙을 돌수 있도록 함."""
        # loop = tqdm(range(self.evaluate_n_episode))

        if self.evaluate_env is not None:
            obs, _ = self.evaluate_env.reset()
        else:
            obs, _ = self.env.reset()
        
        lap_len_list = []
        lap_ret_list = []
        is_done_any = False

        for lap_n in range(self.evaluate_n_episode):
            lap_len = 0
            lap_ret = 0.
            episode_success = False
            while True:

                obs = torch.Tensor(obs).to(self.device)
                act = self.algorithm.get_action(obs=obs)
                if torch.is_tensor(act):
                    act = act.detach().cpu().numpy()
                act = np.squeeze(act)

                if self.evaluate_env is not None:
                    next_obs, reward, terminated, truncated, _ = self.evaluate_env.step(action=act)
                else:
                    next_obs, reward, terminated, truncated, _ = self.env.step(action=act)
                obs = next_obs
                lap_len += 1
                lap_ret += reward

                if terminated or truncated:
                    episode_success = truncated
                    break

            lap_len_list.append(lap_len)
            lap_ret_list.append(lap_ret)
            is_done_any |= episode_success

        return lap_len_list, lap_ret_list, is_done_any

    def evaluate(self): # -> tuple([Sequence[int], Sequence[float], bool]):
        # loop = tqdm(range(self.evaluate_n_episode))

        ep_len_list = []
        ep_ret_list = []
        is_done_any = False

        for episode_n in range(self.evaluate_n_episode):
            obs, _ = self.evaluate_env.reset() if self.evaluate_env is not None else self.env.reset()
            ep_len = 0
            ep_ret = 0. #return of the reward values for the specific episode
            episode_success = False

            while True:
                
                obs = torch.Tensor(obs).to(self.device)
                act = self.algorithm.get_action(obs=obs)
                if torch.is_tensor(act):
                    act = act.detach().cpu().numpy()
                act = np.squeeze(act)
                if self.evaluate_env is not None:
                    next_obs, reward, terminated, truncated, _ = self.evaluate_env.step(action=act)
                else:
                    next_obs, reward, terminated, truncated, _ = self.env.step(action=act)
                obs = next_obs
                ep_len += 1
                ep_ret += reward

                if terminated or truncated:
                    episode_success = truncated
                    break

            ep_len_list.append(ep_len)
            ep_ret_list.append(ep_ret)
            is_done_any |= episode_success

        return ep_len_list, ep_ret_list, is_done_any
        
    def warmup(self, obs: np.ndarray):
        step = 0
        while len(self.buffer) < self.start_step: #self.buffer.cur_len
            step += 1
            if self.warmup_with == "random":
                action = self.env.action_space.sample()
            elif self.warmup_with == "policy":
                action = self.algorithm.get_action(obs=torch.tensor(obs, dtype=torch.float32, device=self.device))
            
            else:
                raise ValueError(f"Invalid warmup with {self.warmup_with}")
            
            # if isinstance(action, torch.Tensor):
            if torch.is_tensor(action):
                action = action.detach().cpu().numpy()

            next_obs, reward, terminated, truncated, info = self.env.step(action=action) 

            experience = Experience.create(obs=obs, action=action, reward=reward, terminated=terminated, truncated=truncated, next_obs=next_obs) #한번에 하나씩..
            
            self.buffer.add(sample=experience) #vectorized env인 경우에만 add_batch로 처리 (병렬적으로 여러개의 environment가 실행 중이기 때문이다.)
  
            
            if np.any(terminated) or np.any(truncated):
                obs, _ = self.env.reset()
            else:
                obs = next_obs

        return obs
    
    
    def add_scalar(self, tag: str, value: float, step: int):
        if step not in self.log_episode_dict:
            self.log_episode_dict[step] = dict()

        self.log_episode_dict[step][tag] = value
        wandb.log({tag: value}, step=step)
        self.logger.add_scalar(tag, value, step)
        self.logger.flush()

    def add_hist(self, info_hist, step):
        for tag, value in info_hist.items():
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            self.logger.add_histogram(tag, np.array(value), step)
            wandb.log({tag: wandb.Histogram(np.array(value))}, step=step)
        self.logger.flush()
       
    def sample(self, obs:np.ndarray):
        action = self.algorithm.get_action(torch.tensor(obs, dtype=torch.float32, device=self.device))
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()
        action = np.squeeze(action)
        # next_obs, reward, terminated, truncated, _ = self.env.step(action=action.detach().cpu().numpy())
        next_obs, reward, terminated, truncated, _ = self.env.step(action)

        experience = Experience.create(obs=obs, action=action, reward=reward, terminated=terminated,
                                       truncated=truncated, next_obs=next_obs, info=None)
        self.buffer.add(sample=experience)

        any_done = self.sl.add(reward=reward, terminated=terminated, truncated=truncated)

        if any_done:
            if self.sample_log_interval.check(step=self.sl.sample_episode):
                self.ul.log(self.add_scalar) #episode return, length 등 episode하나 끝날 때마다 정보 update
            self.progress.update(self.sl.sample_step - self.progress.n) #tqdm progress bar update#

            obs, _ = self.env.reset()
        else:
            obs = next_obs
        
        return obs

    def setup(self, dummy_data:Experience):
        #function called when initializing the Trainer object before starting the training loop#
        self.algorithm.warmup(data=dummy_data)
        self.logger = SummaryWriter(str(self.log_path))
        self.progress = tqdm(total=self.total_step, desc="Sample Step", disable=None)

    def update(self): 
        data = self.buffer.sample(size=self.batch_size)
        info, dist_info = self.algorithm.update(data=data) #SAC에서는 dist_info는 우선 비어 있음 (SDAC에는 있음)

        self.ul.add(info)

        if self.ul.update_step % self.update_log_n_step == 0:
            self.add_hist(dist_info, self.ul.update_step * 5)
            self.ul.log(self.add_scalar)

    def run(self):
        try:
            self.train()
        except KeyboardInterrupt:
            pass
        finally:
            self.finish()
    
    def finish(self):
        self.env.close()
        self.algorithm.save(self.log_path / "state.pkl") #all final state dicts#
        if self.hparams is not None and len(self.last_metrics) > 0:
            exp, ssi, sei = hparams(self.hparams, self.last_metrics)
            #log to tensorboard
            self.logger.file_writer.add_summary(exp)
            self.logger.file_writer.add_summary(ssi)
            self.logger.file_writer.add_summary(sei)
        self.logger.close()
        self.progress.close() #close tqdm

    def train(self):
        obs, _ = self.env.reset(seed = self.seed) #numpy observation state
        # breakpoint()
        obs = self.warmup(obs) #warmup for self.start_step times

        while self.sl.sample_step <= self.total_step:

            for i in range(self.sample_per_iteration):
                obs = self.sample(obs=obs) #next_obs if not done, reset obs if else

            for i in range(self.update_per_iteration):
                self.update()

            if self.save_policy_interval.check(step=self.sl.sample_step):
                policy_pkl_name = self.policy_pkl_template.format(
                    sample_step=self.sl.sample_step,
                    update_step=self.ul.update_step,
                )
                self.algorithm.save_policy(self.log_path / policy_pkl_name) #pathlib Path이기 때문에 경로 이어붙이기가 os.path.join 없이 가능
            
            #run evaluation for specific interval steps#
            if self.eval_policy_interval.check(step=self.sl.sample_step):
                if self.is_racedemia:
                    ep_len_arr, ep_ret_arr, is_done_any = self.evaluate_track()
            
                else:
                    ep_len_arr, ep_ret_arr, is_done_any = self.evaluate()

                if is_done_any:
                    if self.is_racedemia:
                        print(f"Successfully Ended the Lap on Nam Track")
                    state_pkl_name = self.policy_pkl_template.format(sample_step=self.sl.sample_step, update_step=self.ul.update_step)
                    state_pkl_name = "success_eval_state_" + state_pkl_name
                    self.algorithm.save(path=self.log_path / state_pkl_name)
                    info_pkl_name = self.policy_pkl_template.format(sample_step=self.sl.sample_step, update_step=self.ul.update_step)
                    info_pkl_name = "success_eval_info_" + info_pkl_name
                    info_pkl_path = self.log_path / info_pkl_name
                    with open(info_pkl_path, 'wb') as w:
                        pickle.dump({'ep_len': ep_len_arr, 'ep_ret': ep_ret_arr}, w)
                    w.close()
            