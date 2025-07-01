import gymnasium as gym

if __name__ == "__main__":
    '''Continuous action space'''
    env = gym.make("Pendulum-v1", render_mode="human")
    n_episodes = 100
    observation, info = env.reset(seed=42)
    
    for n in range(n_episodes):
        env.render()
        
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            observation, info = env.reset()
    
    env.close()
    
    
    