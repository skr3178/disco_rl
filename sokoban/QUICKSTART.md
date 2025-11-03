# Quick Start Guide

## Installation (One Time Setup)

```bash
cd /Users/skr3178/Downloads/math_is_fun
pip install -e .
```

## Usage Pattern

**IMPORTANT**: You must import `__init__` before using `gym.make()` to register the environment.

### Basic Usage

```python
import __init__  # This line is REQUIRED to register the environment
import gymnasium as gym
import time

env = gym.make("MathIsFunSokoban-v0", render_mode="human")
obs, _ = env.reset()

for _ in range(500):
    a = env.action_space.sample()
    obs, r, done, trunc, _ = env.step(a)
    if done or trunc:
        obs, _ = env.reset()
    time.sleep(0.03)

env.close()
```

### Training with Stable-Baselines3

```bash
# Install SB3 (one time)
pip install "stable-baselines3[extra]" torch
```

```python
import __init__  # REQUIRED!
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Create environment
env = gym.make("MathIsFunSokoban-v0")
env = DummyVecEnv([lambda: env])

# Train with CNN policy (for image observations)
model = PPO("CnnPolicy", env, verbose=1)  # Use CnnPolicy, NOT MlpPolicy!
model.learn(total_timesteps=100000)

# Save model
model.save("sokoban_ppo")

# Test trained agent
import __init__
env_test = gym.make("MathIsFunSokoban-v0", render_mode="human")
obs, _ = env_test.reset()

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env_test.step(int(action))
    if done or truncated:
        print("Episode finished!")
        obs, _ = env_test.reset()

env_test.close()
```

## Key Points

✅ **DO**: Import `__init__` at the start of your script  
✅ **DO**: Use `CnnPolicy` for Stable-Baselines3 (images, not vectors)  
✅ **DO**: Run `pip install -e .` after cloning the repo  

❌ **DON'T**: Forget to import `__init__` (environment won't be registered)  
❌ **DON'T**: Use `MlpPolicy` with Stable-Baselines3 (won't work with images)  

## Verify Installation

```bash
python verify_install.py
```

Should show all green checkmarks ✅

## Examples

- `example_basic.py` - Basic random agent with rendering
- `example_stable_baselines3.py` - RL training with periodic rendering (every 100 steps)
- `example_train_with_continuous_render.py` - RL training with continuous rendering (every step)
- `view_level.py 1` - View a specific level
- `play_sokoban.py` - Play through all levels

### Training with Rendering

**Option 1: Periodic Rendering** (recommended - faster training)
```bash
python example_stable_baselines3.py
```
Shows progress every 100 steps.

**Option 2: Continuous Rendering** (visual but slower)
```bash
python example_train_with_continuous_render.py
```
Shows every single step the agent takes.

## Troubleshooting

**Q: "Environment `MathIsFunSokoban` doesn't exist"**  
A: Add `import __init__` at the top of your script

**Q: "No module named 'math_is_fun'"**  
A: Run `pip install -e .` from the project directory

**Q: Stable-Baselines3 training fails**  
A: Make sure you're using `CnnPolicy` not `MlpPolicy`

## Environment Details

- **Observation**: RGB images (440×638×3)
- **Actions**: 4 discrete (up, right, down, left)
- **Reward**: +1.0 for solving, -0.01 per step
- **Levels**: 60 Sokoban puzzles
- **Episode Length**: 300 steps max

