# Installation Guide

## Quick Start

```bash
# Create conda environment
conda create -n sokoban python=3.10.19 -y
conda activate sokoban

# Install package
cd /path/to/sokoban_rl/sokoban
pip install -e .

# Optional: Install stable-baselines3 for RL training
pip install "stable-baselines3[extra]" torch
```

## Basic Usage

```python
import gymnasium as gym
import mathisfun_sokoban  # Must import first to register environment

env = gym.make("MathIsFunSokoban-v0", render_mode="human")
obs, _ = env.reset()
env.render()

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    
    if done or truncated:
        obs, _ = env.reset()
        env.render()

env.close()
```

## Load Specific Level

```python
import gymnasium as gym
import mathisfun_sokoban
import time

env = gym.make("MathIsFunSokoban-v0", render_mode="human")

# Load level 5 (use 0-59 for levels 1-60)
obs, _ = env.reset(options={"level_idx": 0})
env.render()

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    
    if done:
        obs, _ = env.reset(options={"level_idx": 4})
        env.render()
    elif truncated:
        obs, _ = env.reset(options={"level_idx": 4})
        env.render()
    
    time.sleep(0.05)

env.close()
```

## Command Line Scripts

```bash
# View a specific level (1-60)
python view_level.py 5
```

## Verify Installation

```bash
python example_basic.py
```

## Troubleshooting

**Environment not found:**
- Import `mathisfun_sokoban` before `gym.make()`
- Run `pip install -e .` from sokoban directory

**Window not showing:**
- Call `env.render()` after `env.reset()` and after each `env.step()`
- Install pygame: `pip install pygame`
