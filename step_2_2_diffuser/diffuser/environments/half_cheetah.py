import os
import numpy as np
import gym
from gym import utils
from gym.envs.mujoco import mujoco_env

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0
}
# class HalfCheetahFullObsEnv(mujoco_env.MujocoEnv, utils.EzPickle):
class HalfCheetahFullObsEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 125
    }
    def __init__(self, **kwargs):
        asset_path = os.path.join(
            os.path.dirname(__file__), 'assets/half_cheetah.xml')
         
        self._forward_reward_weight = 1.
        self._ctrl_cost_weight = 0.1
        self._reset_noise_scale = 0.1
        self._exclude_current_positions_from_observation = (False)

        obs_shape = 17 if self._exclude_current_positions_from_observation  else 18
        observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )
        mujoco_env.MujocoEnv.__init__(self, model_path=asset_path, frame_skip=5, observation_space=observation_space, **kwargs)
        utils.EzPickle.__init__(self, self._forward_reward_weight, self._ctrl_cost_weight, self._reset_noise_scale, self._exclude_current_positions_from_observation, 
                                **kwargs)
        
    def control_cost(self, a):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(a))
        return control_cost
    
    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        xvel = (xposafter - xposbefore) / self.dt

        ctrl_cost = self.control_cost(a=action)
        forward_reward = self._forward_reward_weight * xvel

        ob = self._get_obs()

        reward = forward_reward - ctrl_cost
        terminated = False
        info = {
            'x_position': xposafter,
            'x_velocity': xvel,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost
        }

        if self.render_mode == "human":
            self.render()
        return ob, reward, terminated, False, info
    
        # reward_ctrl = - 0.1 * np.square(action).sum()
        # reward_run = (xposafter - xposbefore)/self.dt
        # reward = reward_ctrl + reward_run
        # done = False
        # return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        pos = self.data.qpos.flat.copy()
        vel = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            pos = pos[1:]
        obs = np.concatenate((pos, vel)).ravel()
        return obs

    def reset_model(self):
        noise_low, noise_high = -self._reset_noise_scale, self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )

        qvel = (
            self.init_qvel + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        # self._set_state(qpos, qvel)
        self.set_state(qpos, qvel)
        obs = self._get_obs()
        return obs

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)


    # def reset_model(self):
    #     qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
    #     qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
    #     self.set_state(qpos, qvel)
    #     return self._get_obs()

    # def viewer_setup(self):
    #     self.viewer.cam.distance = self.model.stat.extent * 0.5

    def set(self, state):
        qpos_dim = self.sim.data.qpos.size
        qpos = state[:qpos_dim]
        qvel = state[qpos_dim:]
        self.set_state(qpos, qvel)
        return self._get_obs()
