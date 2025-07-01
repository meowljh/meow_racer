import os
import numpy as np
import gym
from gym import utils
from gym.envs.mujoco import mujoco_env


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}

class Walker2dFullObsEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 125,
    }
    def __init__(self, **kwargs):
        asset_path = os.path.join(
            os.path.dirname(__file__), 'assets/walker2d.xml')
        self._forward_reward_weight = 1.
        self._ctrl_cost_weight = 1e-3
        self._healthy_reward = 1.
        self._terminate_when_unhealthy = True
        self._healthy_z_range = (0.8, 2.0)
        self._healthy_angle_range = (-1., 1.)
        self._reset_noise_scale = 5e-3
        self._exclude_current_positions_from_observation = True

        obs_shape = 18

        if self._exclude_current_positions_from_observation:
            obs_shape -= 1

        observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )

        mujoco_env.MujocoEnv.__init__(self, asset_path, 4, observation_space=observation_space, **kwargs)
        utils.EzPickle.__init__(self)

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy) * self._healthy_reward
        )
    
    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost
    
    @property
    def is_healthy(self):
        z, angle = self.data.qpos[1:3]

        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle
        is_healthy = healthy_z and healthy_angle

        return is_healthy
    
    @property
    def is_terminated(self):
        return not self.is_healthy if self._terminate_when_unhealthy else False
    
    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos.flat.copy()
        qvel = self.sim.data.qvel.flat.copy()
        qvel = np.clip(qvel, -10, 10)

        if self._exclude_current_positions_from_observation:
            qpos = qpos[1:]

        return np.concatenate([qpos, qvel]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
