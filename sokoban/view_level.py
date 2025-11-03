#!/usr/bin/env python3
"""
View a specific Sokoban level
Usage: python view_level.py [level_number]
"""
import sys
import gymnasium as gym
import time
import mathisfun_sokoban  # This auto-registers the environment

# Get level number from command line (default to 0)
level_idx = int(sys.argv[1]) - 1 if len(sys.argv) > 1 else 0
if level_idx < 0:
    level_idx = 0
elif level_idx >= 60:
    level_idx = 59

print(f"Loading Level {level_idx + 1} of 60...")

# Create environment
env = gym.make("MathIsFunSokoban-v0", render_mode="human")
obs, _ = env.reset(options={"level_idx": level_idx})

print(f"✓ Level {level_idx + 1} loaded!")
print(f"✓ Window size: {obs.shape[1]}x{obs.shape[0]} pixels")
print("\nControls:")
print("  - Arrow keys would control the player (if implemented)")
print("  - Currently: Random agent plays automatically")
print("  - Close window or press Ctrl+C to stop\n")

print("Starting in 2 seconds...")
time.sleep(2)

step = 0
try:
    while True:
        # Random action (0=up, 1=right, 2=down, 3=left)
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        
        # Check if window was closed
        if env.unwrapped.window_closed:
            print("\n✓ Window closed by user")
            break
        
        step += 1
        
        if done:
            print(f"🎉 Level {level_idx + 1} solved in {step} steps!")
            print("Restarting level in 3 seconds...")
            time.sleep(3)
            obs, _ = env.reset(options={"level_idx": level_idx})
            step = 0
        elif truncated:
            print(f"⏱️  Level {level_idx + 1} truncated at {step} steps")
            print("Restarting level...")
            time.sleep(1)
            obs, _ = env.reset(options={"level_idx": level_idx})
            step = 0
        
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\n\n✓ Stopped by user")
finally:
    env.close()
    print("✓ Environment closed")

