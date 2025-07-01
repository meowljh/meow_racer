import pickle
import gym
import numpy as np

ant = pickle.load(open('ant_dict.pkl', 'rb'))

print(np.isnan(ant['X']).sum())

print(np.where(ant['means'] == 0)[0])
 
print(np.where(ant['stds'] == 0)[0])
normed = (ant['X'] - ant['means']) / ant['stds']



# from gym.envs.mujoco import mujoco_rendering
# from gym.envs.mujoco.mujoco_rendering import RenderContextOffscreen
# obs = pickle.load(open('obs.pkl', 'rb'))['observation']
# img = pickle.load(open('img.pkl', 'rb'))

# import matplotlib.pyplot as plt
# print(img.max(), img.min())
# plt.imshow(img)
# plt.show()
# import gymnasium as gym
 
# import mujoco_py
# import os
# mj_path = mujoco_py.utils.discover_mujoco()
# xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
# model = mujoco_py.load_model_from_path(xml_path)
# sim = mujoco_py.MjSim(model)

# print(sim.data.qpos)
# # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

# sim.step()
# print(sim.data.qpos)
# # [-2.09531783e-19  2.72130735e-05  6.14480786e-22 -3.45474715e-06
# #   7.42993721e-06 -1.40711141e-04 -3.04253586e-04 -2.07559344e-04
# #   8.50646247e-05 -3.45474715e-06  7.42993721e-06 -1.40711141e-04
# #  -3.04253586e-04 -2.07559344e-04 -8.50646247e-05  1.11317030e-04
# #  -7.03465386e-05 -2.22862221e-05 -1.11317030e-04  7.03465386e-05
# #  -2.22862221e-05]


# import gym
# import gymnasium as gym
# env = gym.make("hopper-medium-expert-v2")
# env = gym.make("Hopper-v4",render_mode="human")
# env.reset()
# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample())
# env.close()