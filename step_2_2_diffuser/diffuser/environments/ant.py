import os
import numpy as np
import gym
from gym import utils
from gym.envs.mujoco import mujoco_env

'''
    qpos : 15
    qvel : 14
    0-2: root x, y, z
    3-7: root quat
    7  : front L hip
    8  : front L ankle
    9  : front R hip
    10 : front R ankle
    11 : back  L hip
    12 : back  L ankle
    13 : back  R hip
    14 : back  R ankle

'''

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0
}

class AntFullObsEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }
    def __init__(self, **kwargs):
        asset_path = os.path.join(
            os.path.dirname(__file__), 'assets/ant.xml')
        self._ctrl_cost_weight = 0.5
        self._use_contact_forces = (True)
        self._contact_cost_weight = 5e-4
        self._healthy_reward = 1.
        self._terminate_when_unhealthy = (True)
        self._healthy_z_range = (0.2, 1.0)
        self._contact_force_range = (-1., 1.)
        self._reset_noise_scale = 0.1
        self._exclude_current_positions_from_observation = (True)

        utils.EzPickle.__init__(self, asset_path, **kwargs)

        obs_shape = 27
        if not self._exclude_current_positions_from_observation:
            obs_shape += 2 # 29
        if self._use_contact_forces:
            obs_shape += 84 # 111

        observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )

        mujoco_env.MujocoEnv.__init__(
            self, asset_path, 5, observation_space=observation_space, **kwargs
        )

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces)
        )
        return contact_cost

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
        return is_healthy

    @property
    def terminated(self):
        terminated = not self.is_healthy if self._terminate_when_unhealthy else False
        return terminated

    # def step(self, action):
    #     xy_position_before = self.get_body_com("torso")[:2].copy()
    #     self.do_simulation(action, self.frame_skip)
    #     xy_position_after = self.get_body_com("torso")[:2].copy()

    #     xy_velocity = (xy_position_after - xy_position_before) / self.dt
    #     x_velocity, y_velocity = xy_velocity

    #     forward_reward = x_velocity
    #     healthy_reward = self.healthy_reward

    #     rewards = forward_reward + healthy_reward

    #     costs = ctrl_cost = self.control_cost(action)

    #     terminated = self.terminated
    #     observation = self._get_obs()
    #     info = {
    #         "reward_forward": forward_reward,
    #         "reward_ctrl": -ctrl_cost,
    #         "reward_survive": healthy_reward,
    #         "x_position": xy_position_after[0],
    #         "y_position": xy_position_after[1],
    #         "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
    #         "x_velocity": x_velocity,
    #         "y_velocity": y_velocity,
    #         "forward_reward": forward_reward,
    #     }
    #     if self._use_contact_forces:
    #         contact_cost = self.contact_cost
    #         costs += contact_cost
    #         info["reward_ctrl"] = -contact_cost

    #     reward = rewards - costs

    #     if self.render_mode == "human":
    #         self.render()
    #     return observation, reward, terminated, False, info

    # def _get_obs(self):
    #     position = self.data.qpos.flat.copy()
    #     velocity = self.data.qvel.flat.copy()

    #     if self._exclude_current_positions_from_observation:
    #         position = position[2:]

    #     if self._use_contact_forces:
    #         contact_force = self.contact_forces.flat.copy()
    #         return np.concatenate((position, velocity, contact_force))
    #     else:
    #         return np.concatenate((position, velocity))

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return (
            ob,
            reward,
            done,
            False,
            dict(
                reward_forward=forward_reward,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )

    def _get_obs(self):

        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        contact_force = self.contact_forces.flat.copy()
        contact_force = np.clip(
            contact_force,
            self._contact_force_range[0],
            self._contact_force_range[1]
        )
        if self._exclude_current_positions_from_observation:
            position = position[2:]
 
        if self._use_contact_forces:
            return np.concatenate((position, velocity, contact_force))
        
        return np.concatenate((position, velocity))

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, 
            low=noise_low, 
            high=noise_high
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * self._rest_noise_scale
        self.set_state(qpos, qvel)
        obs = self._get_obs()
        
        return obs