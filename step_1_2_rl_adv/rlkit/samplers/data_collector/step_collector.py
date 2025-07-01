from collections import deque, OrderedDict

import numpy as np

from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.data_management.path_builder import PathBuilder
from rlkit.samplers.data_collector.base import StepCollector


class MdpStepCollector(StepCollector):
    def __init__(
            self,
            env,
            policy,
            
            plotter,
            
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
            
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._plotter = plotter
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs

        self._num_steps_total = 0
        self._num_paths_total = 0
        self._obs = None  # cache variable
         

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._obs = None

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        return dict(
            env=self._env,
            policy=self._policy,
        )

    def collect_new_steps(
            self,
            max_path_length:int,
            num_steps:int,
            discard_incomplete_paths:bool,
            # initial_data_collect:bool=False,
            # do_warmup:bool=False
    ):
        for _ in range(num_steps):
            self.collect_one_step(
                max_path_length=max_path_length,
                discard_incomplete_paths=discard_incomplete_paths
            )
            # if num_steps == 1:
                # breakpoint()
            # if _ == num_steps-1 and num_steps != 1:
            #     self.collect_new_steps(max_path_length, discard_incomplete_paths, initial_data_collect, 
            #                            do_warmup=do_warmup)
            # else:
            #     self.collect_one_step(max_path_length, discard_incomplete_paths, False,
            #                           do_warmup=do_warmup)
            # if num_steps == 1:
                # breakpoint()
        # breakpoint()
        
        """
        초기에 replay buffer에 training 데이터가 쌓여야 하는 경우에, terminate된 데이터가 아니라고 해도 데이터를 추가해 주어야 함.
        """
        # if len(self._num_steps_total) == 0:
            # breakpoint()
        if self._num_steps_total == 0:
            self._handle_rollout_ending(max_path_length,
                                        discard_incomplete_paths)
            # for _ in range(max_path_length - num_steps + 1):
            #     self.collect_one_step(
            #         max_path_length=max_path_length,
            #         discard_incomplete_paths=discard_incomplete_paths
            #     ) 
                
    def _warmup_actor(self):
        action_space = self._env.unwrapped.action_space
        action_dim = self._env.unwrapped.environment_config['action']['action_dim']
        ac = self._env.unwrapped.agent_config
        if action_dim == 2:
            warmup_min = [ac['warmup_steer_min'], ac['warmup_throttle_min']]
            warmup_max = [ac['warmup_steer_max'], ac['warmup_throttle_max']]
        else:
            warmup_min = [ac['warmup_steer_min'], ac['warmup_aps_min'], ac['warmup_bps_min']]
            warmup_max = [ac['warmup_steer_max'], ac['warmup_aps_max'], ac['warmup_bps_max']]
            
        action = []
        for i, (m, M) in enumerate(zip(action_space.low, action_space.high)):
            if i == 0:
                # steer = np.random.uniform(-0.1, 0.1)
                steer = np.random.uniform(warmup_min[i], warmup_max[i])
                action.append(steer)
            elif i == 1:
                if action_dim == 2:
                    throttle = np.random.uniform(0., 1.)
                    
                else:
                    # throttle = np.random.uniform(0.4, 0.8)
                    throttle = np.random.uniform(warmup_min[i], warmup_max[i])
                
                action.append(throttle)
            else:
                if action_dim == 3:
                    # brake = np.random.uniform(-1., -0.7)
                    brake = np.random.uniform(warmup_min[i], warmup_max[i])
                action.append(brake)
        
        return np.array(action)
            
            
    def collect_one_step(
            self,
            max_path_length,
            discard_incomplete_paths,
            # do_warmup:bool=False
    ):
        if self._obs is None:
            self._start_new_rollout()
            
        
        # if do_warmup:
        #     action = self._warmup_actor()
        #     agent_info = {}
        # else:
        #     action, agent_info = self._policy.get_action(self._obs)
        # if self._env.unwrapped.use_style:
        #     action, agent_info = self._policy.get_action_style(self._obs, self._env.unwrapped.style_level)
        # else:
        #     action, agent_info = self._policy.get_action(self._obs)
        # breakpoint()
        # action, agent_info = self._policy.get_action_style(self._obs)
        action, agent_info = self._policy.get_action(self._obs)
        env_step_ret = self._env.step(action)
        if len(env_step_ret) == 4:
            next_ob, reward, terminal, env_info = env_step_ret
        elif len(env_step_ret) == 5:
            next_ob, reward, terminated, truncated, env_info = env_step_ret
            terminal = terminated | truncated

        # terminal = terminated | truncated
        if self._render:
            self._env.render(**self._render_kwargs)
        terminal = np.array([terminal])
        reward = np.array([reward])
        
        self._plotter._log_action(action)
        self._plotter._log_reward(reward)
        self._plotter._update_global_step()
        
        # store path obs
        self._current_path_builder.add_all(
            observations=self._obs,
            actions=action,
            rewards=reward,
            next_observations=next_ob,
            terminals=terminal,
            agent_infos=agent_info,
            env_infos=env_info,
        ) 
        ##하나의 새로운 episode, 즉 새로운 track을 탐색을 시작하고 한 episode의 max_path_length를 초과하면 episode reset ##
        #지정해 둔 max_path_length가 짧다 보니까 끝까지 하나의 track에 대해서 진척을 못함
        if terminal or len(self._current_path_builder) >= max_path_length: # or initial_data_collect: 
            self._handle_rollout_ending(max_path_length,
                                        discard_incomplete_paths)
            self._start_new_rollout()
    
            # self._plotter._reset_rollout()
        else:
            self._obs = next_ob

    def _start_new_rollout(self):
        self._current_path_builder = PathBuilder()
        self._obs, info = self._env.reset()
         
    def _handle_rollout_ending(
            self,
            max_path_length,
            discard_incomplete_paths
    ):
        if len(self._current_path_builder) > 0:
            path = self._current_path_builder.get_all_stacked()
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ): ##근데 길이도 못채웠고 terminate된것도 아니고 동시에 미완성 path는 replay buffer에 추가를 안하는거면 buffer에 추가가 안됨
                return
            
            self._epoch_paths.append(path)
            self._num_paths_total += 1
            self._num_steps_total += path_len


class GoalConditionedStepCollector(StepCollector):
    
    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
            observation_key='observation',
            desired_goal_key='desired_goal',
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs
        self._observation_key = observation_key
        self._desired_goal_key = desired_goal_key

        self._num_steps_total = 0
        self._num_paths_total = 0
        self._obs = None  # cache variable

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._obs = None

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        return dict(
            env=self._env,
            policy=self._policy,
            observation_key=self._observation_key,
            desired_goal_key=self._desired_goal_key,
        )

    def start_collection(self):
        self._start_new_rollout()

    def end_collection(self):
        epoch_paths = self.get_epoch_paths()
        return epoch_paths

    def collect_new_steps(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        for _ in range(num_steps):
            self.collect_one_step(max_path_length, discard_incomplete_paths)

    def collect_one_step(
            self,
            max_path_length,
            discard_incomplete_paths,
    ):
        if self._obs is None:
            self._start_new_rollout()

        new_obs = np.hstack((
            self._obs[self._observation_key],
            self._obs[self._desired_goal_key],
        ))
        action, agent_info = self._policy.get_action(new_obs)
        env_step_ret = self._env.step(action)
        if len(env_step_ret) == 4:
            next_ob, reward, terminal, env_info = env_step_ret
        elif len(env_step_ret) == 5:
            next_ob, reward, terminated, truncated, env_info = env_step_ret
            terminal = terminated | truncated

        if self._render:
            self._env.render(**self._render_kwargs)
        terminal = np.array([terminal])
        reward = np.array([reward])
        # store path obs
        self._current_path_builder.add_all(
            observations=self._obs,
            actions=action,
            rewards=reward,
            next_observations=next_ob,
            terminals=terminal,
            agent_infos=agent_info,
            env_infos=env_info,
        )
        if terminal or len(self._current_path_builder) >= max_path_length:
            self._handle_rollout_ending(max_path_length,
                                        discard_incomplete_paths)
            self._start_new_rollout()
        else:
            self._obs = next_ob

    def _start_new_rollout(self):
        self._current_path_builder = PathBuilder()
        self._obs = self._env.reset()

    def _handle_rollout_ending(
            self,
            max_path_length,
            discard_incomplete_paths
    ):
        if len(self._current_path_builder) > 0:
            path = self._current_path_builder.get_all_stacked()
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                return
            self._epoch_paths.append(path)
            self._num_paths_total += 1
            self._num_steps_total += path_len


class ObsDictStepCollector(StepCollector):
    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
            observation_key='observation',
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs
        self._observation_key = observation_key

        self._num_steps_total = 0
        self._num_paths_total = 0
        self._obs = None  # cache variable

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._obs = None

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        return dict(
            env=self._env,
            policy=self._policy,
            observation_key=self._observation_key,
        )

    def start_collection(self):
        self._start_new_rollout()

    def end_collection(self):
        epoch_paths = self.get_epoch_paths()
        return epoch_paths

    def collect_new_steps(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        for _ in range(num_steps):
            self.collect_one_step(max_path_length, discard_incomplete_paths)

    def collect_one_step(
            self,
            max_path_length,
            discard_incomplete_paths,
    ):
        if self._obs is None:
            self._start_new_rollout()

        new_obs = self._obs[self._observation_key]
        action, agent_info = self._policy.get_action(new_obs)
        env_step_ret = self._env.step(action)
        if len(env_step_ret) == 4:
            next_ob, reward, terminal, env_info = env_step_ret
        elif len(env_step_ret) == 5:
            next_ob, reward, terminated, truncated, env_info = env_step_ret
            terminal = terminated | truncated

        if self._render:
            self._env.render(**self._render_kwargs)
        terminal = np.array([terminal])
        reward = np.array([reward])
        # store path obs
        self._current_path_builder.add_all(
            observations=self._obs,
            actions=action,
            rewards=reward,
            next_observations=next_ob,
            terminals=terminal,
            agent_infos=agent_info,
            env_infos=env_info,
        )
        if terminal or len(self._current_path_builder) >= max_path_length:
            self._handle_rollout_ending(max_path_length,
                                        discard_incomplete_paths)
            self._start_new_rollout()
        else:
            self._obs = next_ob

    def _start_new_rollout(self):
        self._current_path_builder = PathBuilder()
        self._obs = self._env.reset()

    def _handle_rollout_ending(
            self,
            max_path_length,
            discard_incomplete_paths
    ):
        if len(self._current_path_builder) > 0:
            path = self._current_path_builder.get_all_stacked()
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                return
            self._epoch_paths.append(path)
            self._num_paths_total += 1
            self._num_steps_total += path_len

