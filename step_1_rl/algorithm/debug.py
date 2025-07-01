import gymnasium as gym
import torch
import matplotlib.pyplot as plt

mu = torch.FloatTensor([0.0])
std = torch.FloatTensor([0.1])

for i in range(1000000):
    z = torch.normal(mu, std).numpy()
    plt.scatter(i, z[0])
plt.savefig("normal.png")
# env = gym.make("HalfCheetah-v5", render_mode="human")
# state_space = env.observation_space
# print(state_space.shape)
# action_space = env.action_space
# print(action_space.shape)

# print(isinstance(action_space, gym.spaces.Discrete))
# print(isinstance(action_space, gym.spaces.Box))