import abc
import os, sys
from pathlib import Path
import torch
import numpy as np

import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import (
    PathCollector,
    StepCollector,
)


class OnlineRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: StepCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            
            num_step_for_expl_data_collect=1,
            ###### additional arguments for TESTING on nam-c track ######
            customized_test_fn=None,
            test_env=None,
            test_policy=None,
            test_kwargs=None,
            
            warmup_actor_step:float=-1
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.num_step_for_expl_data_collect = num_step_for_expl_data_collect
        
        self._customized_test_fn = customized_test_fn
        self.test_env = test_env
        self.test_policy = test_policy
        self.test_kwargs = test_kwargs if test_kwargs is not None else {}

        self.warmup_actor_step = warmup_actor_step
        
        
        assert self.num_trains_per_train_loop >= self.num_expl_steps_per_train_loop, \
            'Online training presumes num_trains_per_train_loop >= num_expl_steps_per_train_loop'

    def _train(self):
        self.training_mode(False) 
        # _, test_success = self._customized_test_fn(eval_policy=self.test_policy, 
        #                              eval_env=self.test_env,
        #                              epoch_num=-1,
        #                              **self.test_kwargs)
        print("STARTING INITIAL EXPLORATION before TRAINING")
        #학습 전에 최소한으로 Replay Buffer을 채워야 하기 때문에 exploration 진행
        if self.min_num_steps_before_training > 0:
            self.expl_data_collector.collect_new_steps(
                self.max_path_length,
                self.min_num_steps_before_training, #여러번의 step동안 replay buffer에 저장할 수 있는 데이터를 수집. 근데 빨리 terminate되면 batch size보다 크거나 같은 크기의 데이터를 모으지 못할수도 있음.
                discard_incomplete_paths=False,
                # initial_data_collect=True,
                # do_warmup=False #True(초기에 데이터 모을때는 항상 APS만 밟는 data로 모아주어야 하는 것인가???)
            )
            init_expl_paths = self.expl_data_collector.get_epoch_paths()
            # breakpoint()
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

            gt.stamp('initial exploration', unique=True)
        print("FINISHED INITIAL EXPLORATION before TRAINING")
        # print(f"Number of steps collected in the buffer: {len(self.expl_data_collector.__epoch_paths)}")
        print(self.expl_data_collector.get_diagnostics())
        # breakpoint()
        step_cnt = 0
        num_trains_per_expl_step = self.num_trains_per_train_loop // self.num_expl_steps_per_train_loop
        # breakpoint()
        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            print(f"======== STARTING EPOCH {epoch} ========")
            #step마다 path들을 만드는게, "done" 상태가 될때까지 계속 진행됨#
            ##### evaluation은 오직 logging을 위해서 한다고 하니, 굳이 필요 없을 것 같음 #####
            # print(f"EVALUATING ON RANDOM TRACKS for EPOCH {epoch}")
            # self.eval_data_collector.collect_new_paths(     
            #     self.max_path_length,
            #     self.num_eval_steps_per_epoch,
            #     discard_incomplete_paths=True,
            # )
            # gt.stamp('evaluation sampling')
            ########################################################################################
            #매 epoch마다 testing 결과 저장 및 남양을 완주 한 경우에만 weight등을 저장
            print(f"TESTING ON NAM-C TRACK for EPOCH {epoch}")
            if self.test_env.unwrapped.use_style:
                # style_ranges = np.arange(0, 1+self.test_env.unwrapped.style_step_size, self.test_env.unwrapped.style_step_size)
                style_ranges = self.test_env.unwrapped.style_setting_candidates
                for style in style_ranges:
                    _, test_success = self._customized_test_fn(
                        eval_policy = MakeDeterministic(self.trainer.policy),
                        eval_env=self.test_env,
                        epoch_num=epoch,
                        style=style,
                        **self.test_kwargs
                    )
                    if test_success:
                        test_log_path = f"{self.test_kwargs['test_log_path']}/ckpt/epoch_{epoch}__style_{style}"
                        Path(test_log_path).mkdir(parents=True, exist_ok=True)
                        networks = self.trainer.networks_dict
                        for key, net in networks.items():
                            torch.save(net.state_dict(), f"{test_log_path}/{key}.pth")

                        optimizers = self.trainer.optimizers_dict
                        for key, optim in optimizers.items():
                            torch.save(optim.state_dict(), f"{test_log_path}/{key}.pth")
            else:
                _, test_success = self._customized_test_fn(
                                    #  eval_policy=self.test_policy, 
                                     eval_policy = MakeDeterministic(self.trainer.policy),
                                     eval_env=self.test_env,
                                     epoch_num=epoch,
                                     **self.test_kwargs
                                     )

                if test_success:
                    test_log_path = f"{self.test_kwargs['test_log_path']}/ckpt/epoch_{epoch}"
                    Path(test_log_path).mkdir(parents=True, exist_ok=True)
                    networks = self.trainer.networks_dict
                    for key, net in networks.items():
                        torch.save(net.state_dict(), f"{test_log_path}/{key}.pth")

                    optimizers = self.trainer.optimizers_dict
                    for key, optim in optimizers.items():
                        torch.save(optim.state_dict(), f"{test_log_path}/{key}.pth")

            ## just save parameters for every epochs (for continued training) ##
            test_log_path = f"{self.test_kwargs['test_log_path']}/ckpt/recent"
            Path(test_log_path).mkdir(parents=True, exist_ok=True)
            networks = self.trainer.networks_dict
            for key, net in networks.items():
                torch.save(net.state_dict(), f"{test_log_path}/{key}.pth")
                    
            optimizers = self.trainer.optimizers_dict
            for key, optim in optimizers.items():
                torch.save(optim.state_dict(), f"{test_log_path}/{key}.pth")
            print(f"FINISHED TESTING ON NAM-C TRACK for EPOCH {epoch}")
            ########################################################################################
            #하나의 epoch마다의 training loop을 "exploration data 모으고 parameter update하는 과정"으로 정의함

            for _ in range(self.num_train_loops_per_epoch):
                # exploration steps per each training loop # 
                print(f"STARTING TO TRAIN ON EPOCH {epoch}")
                for _ in range(self.num_expl_steps_per_train_loop):
                    #각 exploration step마다 num_trains_per_expl_step번 만큼 반복해서 sac를 학습
                    # print(f"COLLECTING REPLAY BUFFER FOR SINGLE STEP WITH EXPLORATION ON EPOCH {epoch}")
                    ##근데  1step만을 collect하는데, 여기서 1개의 step마다 데이터를 모으고, exploration step collector의 내부적으로는 deque에 저장이 되어 있을 것임.
              
                    self.expl_data_collector.collect_new_steps(
                        self.max_path_length,
                        # 1,  # num steps
                        self.num_step_for_expl_data_collect,
                        discard_incomplete_paths=False,
                        # initial_data_collect=False,
                        # do_warmup=step_cnt <= self.warmup_actor_step
                    )
                    # breakpoint()
                    step_cnt += 1
                    gt.stamp('exploration sampling', unique=False)
                    
                    
                    self.training_mode(True)
                    '''한번의 exploration step에 대한 학습 횟수인데, 결과적으로는 한번의 step을 추가하고 replay buffer에 추'''
                    for _ in range(num_trains_per_expl_step):
                        train_data = self.replay_buffer.random_batch(
                            self.batch_size)
                        self.trainer.train(np_batch=train_data)
                    gt.stamp('training', unique=False)
                    self.training_mode(False)
                    
                print(f"FINISHED TRAINING ON EPOCH {epoch}")
            
            print(f"ADDING EXPLORED TRAJECTORY PATHS TO REPLAY BUFFER on EPOCH {epoch}")
            new_expl_paths = self.expl_data_collector.get_epoch_paths()
            self.replay_buffer.add_paths(new_expl_paths)
            gt.stamp('data storing', unique=False)
            print(f"FINISHED ADDING EXPLORED TRAJECTORY PATHS TO REPLAY BUFFER on EPOCH {epoch}")

            self._end_epoch(epoch)
