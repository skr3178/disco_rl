#!/usr/bin/env python3
"""
Simple script to play MathIsFun Sokoban with visual rendering
"""
import gymnasium as gym
import time
from gymnasium.envs.registration import register
from mathisfun_sokoban import MathIsFunSokoban

# Register the environment
register(
    id="MathIsFunSokoban-v0",
    entry_point="mathisfun_sokoban:MathIsFunSokoban",
    max_episode_steps=300,
)

# Create environment
print("Creating MathIsFun Sokoban environment...")
env = gym.make("MathIsFunSokoban-v0", render_mode="human")
obs, _ = env.reset()

print(f"✓ Environment created!")
print(f"✓ Window size: {obs.shape[1]}x{obs.shape[0]} pixels")
print(f"✓ Total levels: 60")
print("\nControls:")
print("  - Random agent will play automatically")
print("  - Close the window to stop")
print("  - Or press Ctrl+C\n")

print("Starting game in 2 seconds...")
time.sleep(2)

step = 0
try:
    while True:
        # Random action
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        
        step += 1
        
        if done:
            print(f"🎉 Level solved in {step} steps!")
            obs, _ = env.reset()
            step = 0
            time.sleep(1)  # Pause to see the win
        elif truncated:
            print(f"⏱️  Level truncated at {step} steps")
            obs, _ = env.reset()
            step = 0
        
        time.sleep(0.05)  # Slow down so you can see it

except KeyboardInterrupt:
    print("\n\n✓ Stopped by user")
except Exception as e:
    print(f"\n\n❌ Error: {e}")
finally:
    env.close()
    print("✓ Environment closed")

