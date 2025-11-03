"""
Basic example of using the MathIsFun Sokoban environment
Make sure to install the package first: pip install -e .
"""
import __init__  # This registers the environment
import gymnasium as gym
import time

# Create environment (package must be installed for this to work)
env = gym.make("MathIsFunSokoban-v0", render_mode="human")

obs, _ = env.reset()
env.render()  # Render the initial state

for _ in range(500):
    a = env.action_space.sample()
    obs, r, done, trunc, _ = env.step(a)
    env.render()  # Render after each step to update the display
    
    if done or trunc:
        obs, _ = env.reset()
        env.render()  # Render after reset
    
    time.sleep(0.03)

env.close()

