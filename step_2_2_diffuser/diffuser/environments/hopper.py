import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco import MuJocoPyEnv
from gym.spaces import Box


class HopperFullObsEnv(mujoco_env.MujocoEnv, utils.EzPickle):
# class HopperFullObsEnv(MuJocoPyEnv, utils.EzPickle):
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
            os.path.dirname(__file__), 'assets/hopper.xml')
        
        observation_space = Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float64)


        mujoco_env.MujocoEnv.__init__(self, 
        # MuJocoPyEnv.__init__(self,
                                      model_path=asset_path, 
                                      render_mode="human",
                                      frame_skip=4,
                                      observation_space=observation_space,
                                      **kwargs)
        utils.EzPickle.__init__(self, **kwargs)

        self._healthy_reward = 1.
        self._terminate_when_unhealthy = True
        self._healthy_state_range = (-100., +100.)
        self._healthy_z_range = (0.7, float("inf"))
        self._healthy_angle_range = (-0.2, +0.2)
        self._reset_noise_scale = 5e-3
        self._exclude_current_positions_from_observation = True
        self.canmera_id=0


    def step(self, a):
        posbefore = self.data.qpos[0] # x_pos_before
        self.do_simulation(a, self.frame_skip)
        posafter = self.data.qpos[0] # x_pos_after
        vel = (posafter - posbefore) / self.dt # x_velocity

        ctrl_cost = self.control_cost(a)

        forward_reward = vel
        healthy_reward = self.healthy_reward

        reward = forward_reward + healthy_reward
        costs = ctrl_cost

        observation = self._get_obs()
        reward = reward - costs
        terminated = self.terminated

        info = {"x_position": posafter, "x_velocity": vel}

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info



    # def step(self, a):
    #     ## same as envs/mujoco/hopper.py
    #     posbefore = self.sim.data.qpos[0]
    #     self.do_simulation(a, self.frame_skip)
    #     posafter, height, ang = self.sim.data.qpos[0:3]
    #     alive_bonus = 1.0
    #     reward = (posafter - posbefore) / self.dt
    #     reward += alive_bonus
    #     reward -= 1e-3 * np.square(a).sum()
    #     s = self.state_vector()
    #     done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
    #                 (height > .7) and (abs(ang) < .2))
    #     ob = self._get_obs()
    #     return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            # self.sim.data.qpos.flat[1:],
            self.sim.data.qpos.flat,
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        qpos = self.init_qpos + \
            self.np_random.uniform(
                low=-self._reset_noise_scale, high=self._reset_noise_scale, size=self.model.nq
            )
            # self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + \
            self.np_random.uniform(
                low=-self._reset_noise_scale, high=self._reset_noise_scale, size=self.model.nq
        )
            # self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
    
        return self._get_obs()



    def viewer_setup(self):
        DEFAULT_CAMERA_CONFIG = {
            "trackbodyid": 2,
            "distance": 3.0,
            "lookat": np.array((0.0, 0.0, 1.15)),
            "elevation": -20.0,
        }
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
    # def viewer_setup(self):
    #     self.viewer.cam.trackbodyid = 2
    #     self.viewer.cam.distance = self.model.stat.extent * 0.75
    #     self.viewer.cam.lookat[2] = 1.15
    #     self.viewer.cam.elevation = -20

    # def set(self, state):
    #     qpos_dim = self.sim.data.qpos.size
    #     # qpos_dim = self.model.vis.qpos.size
    #     qpos = state[:qpos_dim]
    #     qvel = state[qpos_dim:]
    #     self.set_state(qpos, qvel)
    #     return self._get_obs()

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
    def is_healthy(self):
        z, angle = self.data.qpos[1:3]
        state = self.state_vector()[2:]

        min_state, max_state = self._healthy_state_range
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_state = np.all(np.logical_and(min_state < state, state < max_state))
        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle

        is_healthy = all((healthy_state, healthy_z, healthy_angle))

        return is_healthy

    @property
    def terminated(self):
        terminated = not self.is_healthy if self._terminate_when_unhealthy else False
        return terminated
