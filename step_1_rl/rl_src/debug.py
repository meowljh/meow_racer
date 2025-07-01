import numpy as np
a = np.zeros((100000, 1, 3))
# env_indices = np.ones(10)
env_indices = np.random.randint(0, 1, size=(10,))
batch_indices = np.random.randint(0, 10, size=(10,))
print(env_indices, env_indices.shape)
b = a[batch_indices, env_indices, :]
print(b)
print(b.shape)
# dones = np.zeros(3)
# dones[1] = 1

# a = (dones == 1)
# print(a.astype(int))
# c = np.zeros((2, 1, 4))
# print(c[0, :, 0])
# c = np.array([
#     [[1], [0, 0, 0, 0]],
#     [[1], [0, 0, 0, 0]]
# ])
# print(c.shape)

# buffer_size = 100
# pos = 3
# n_envs = 3
# obs = np.random.rand(33)
# obs_shape = obs.shape
# observation = np.zeros((buffer_size, 1,  n_envs, *obs_shape))
# ids = np.array([1,2,2])

# new_obs = np.random.rand(n_envs, 33)
# new_obs = np.expand_dims(new_obs, 0) ## (1, 3, 33)
# new_obs = np.concatenate((new_obs, np.reshape(ids, (1, 3, 1))), 0)
# print(new_obs.shape)
# observation[pos] = np.array(new_obs)

