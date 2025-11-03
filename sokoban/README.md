# MathIsFun Sokoban - Gymnasium Environment

A complete Gymnasium-compatible environment featuring all 60 Sokoban puzzle levels from [MathsIsFun.com](https://www.mathsisfun.com/games/sokoban.html).

![Sokoban Game](https://img.shields.io/badge/Levels-60-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Gymnasium](https://img.shields.io/badge/Gymnasium-Compatible-orange)

## 🎮 Overview

Sokoban is a classic Japanese puzzle game where you push boxes onto target spots. This implementation provides:
- ✅ All 60 official levels from MathsIsFun.com
- ✅ Beautiful 3D-style graphics with proper shading
- ✅ Full Gymnasium API compatibility
- ✅ Ready for RL training
- ✅ Visual rendering with pygame

## 📦 Installation

### Prerequisites
```bash
conda create -n sokoban python=3.10
conda activate sokoban
```

### Install the Package (Recommended)
```bash
cd /Users/skr3178/Downloads/math_is_fun
pip install -e .
```

This installs the environment as a Python package, allowing you to use `gym.make("MathIsFunSokoban-v0")` from anywhere.

### Or Install Dependencies Only
```bash
pip install gymnasium numpy pygame Pillow
```

Note: Without installing the package, you'll need to manually import the environment or register it (see examples below).

## 🚀 Quick Start

### Train with Stable-Baselines3
```bash
pip install -e .
pip install "stable-baselines3[extra]" torch
python example_stable_baselines3.py  # Opens rendering window
```

See **[TRAINING.md](TRAINING.md)** for complete training guide.

### View Levels
```bash
python view_level.py 1      # View Level 1 (easiest)
python view_level.py 30     # View Level 30 (medium)
python view_level.py 60     # View Level 60 (hardest)
```

### 3. Use in Your Code

**Option A: With Package Installed** (after running `pip install -e .`)
```python
import __init__  # This registers the environment with Gymnasium
import gymnasium as gym
import time

# Create environment - now it's registered!
env = gym.make("MathIsFunSokoban-v0", render_mode="human")
obs, _ = env.reset()

for _ in range(500):
    action = env.action_space.sample()  # Random action
    obs, reward, done, truncated, _ = env.step(action)
    
    if done or truncated:
        obs, _ = env.reset()
    
    time.sleep(0.03)

env.close()
```

**Option B: Direct Import** (without installing package)
```python
from math_is_fun.mathisfun_sokoban import MathIsFunSokoban

env = MathIsFunSokoban(render_mode="human")
obs, _ = env.reset()

for _ in range(500):
    action = env.action_space.sample()
    obs, reward, done, truncated, _ = env.step(action)
    
    if done or truncated:
        obs, _ = env.reset()

env.close()
```

### 4. Load a Specific Level
```python
# Reset to a specific level (0-59)
obs, info = env.reset(options={"level_idx": 10})
```

## 🎯 Environment Details

### Action Space
- **Type**: `Discrete(9)`
- **Actions**:
  - `0` - No Operation
  - `1` - Push Up
  - `2` - Push Down
  - `3` - Push Left
  - `4` - Push Right
  - `5` - Move Up
  - `6` - Move Down
  - `7` - Move Left
  - `8` - Move Right

### Observation Space
- **Type**: `Box(0, 255, shape=(height, width, 3), dtype=uint8)`
- **Format**: RGB image of the game state
- **Size**: Approximately 440×638 pixels (varies by level)

### Rewards
- `-0.1` - Per step penalty (encourages efficiency)
- `+1.0` - Push box onto target
- `-1.0` - Push box off target
- `+10.0` - Level solved (all boxes on targets)

### Episode Termination
- **Done**: All boxes placed on target spots
- **Truncated**: 300 steps reached (configurable)

## 🎨 Visual Elements

| Element | Color | Description |
|---------|-------|-------------|
| 🧱 Walls | Dark Gray | Impassable barriers |
| 🟫 Boxes | Brown (3D) | Objects to push onto targets |
| 🟩 Targets | Green Circles | Goal positions for boxes |
| 🔴 Player | Red Circle | Your character |
| ✨ Box on Target | Golden Brown | Box correctly placed |
| 🟦 Background | Gray | Empty space outside level |

## 📁 Project Structure

```
math_is_fun/
├── README.md                               # Overview & quick start
├── TRAINING.md                             # ⭐ Training guide (START HERE)
├── QUICKSTART.md                           # Quick reference
├── setup.py                                # Package setup
├── __init__.py                             # Environment registration
├── mathisfun_sokoban.py                   # Main environment
├── levels_clean.json                      # 60 Sokoban levels
├── example_stable_baselines3.py           # ⭐ Train with rendering
├── example_train_with_continuous_render.py # Train with continuous rendering
├── test_render_callback.py                # Quick rendering test
├── verify_install.py                      # Verify installation
├── view_level.py                          # View levels
└── play_sokoban.py                       # Play manually
```

## 🎓 Training RL Agents

**See [TRAINING.md](TRAINING.md) for complete guide.**

### Quick Training
```bash
pip install -e .
pip install "stable-baselines3[extra]" torch
python example_stable_baselines3.py  # Trains with rendering
```

### Minimal Code
```python
import __init__
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

env = gym.make("MathIsFunSokoban-v0")
env = DummyVecEnv([lambda: env])

model = PPO("CnnPolicy", env, verbose=1)  # Use CnnPolicy for images!
model.learn(total_timesteps=100000)
model.save("sokoban_model")
```

**Key Points:**
- ✅ Use `CnnPolicy` (not MlpPolicy) - environment outputs images
- ✅ Import `__init__` first to register environment
- ✅ Call `env.render()` after `step()` to see visualization

**Examples:**
- `example_stable_baselines3.py` - Training with rendering (every 100 steps)
- `example_train_with_continuous_render.py` - Continuous rendering (every step)
- `test_render_callback.py` - Quick test (500 steps)

## 🎮 Controls & Features

### Current Features
- ✅ Visual rendering with pygame
- ✅ 60 progressively challenging levels
- ✅ Proper Sokoban game mechanics
- ✅ Episode tracking and statistics
- ✅ Window close detection (click X to exit)

### Keyboard Controls
Currently, the environment uses random actions. To implement keyboard controls:

```python
import pygame

# In your game loop
for event in pygame.event.get():
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_UP:
            action = 0
        elif event.key == pygame.K_RIGHT:
            action = 1
        elif event.key == pygame.K_DOWN:
            action = 2
        elif event.key == pygame.K_LEFT:
            action = 3
```

## 📊 Level Statistics

- **Total Levels**: 60
- **Difficulty**: Progressive (1 = easiest, 60 = hardest)
- **Smallest Level**: 8×8 grid
- **Largest Level**: ~29×27 grid
- **Average Boxes**: 4-8 per level
- **Source**: [MathsIsFun.com/games/sokoban.html](https://www.mathsisfun.com/games/sokoban.html)

## 🐛 Troubleshooting

### Window doesn't appear
```bash
# Test pygame installation
python test_pygame.py

# If that works, test the environment
python test_env.py
```

### Window doesn't close when clicking X
- Updated in latest version - should work now
- Alternatively, press `Ctrl+C` in terminal

### Graphics are too small
- Window size is set to 640px (4x scale)
- Edit `self.scale = 640 // max(max_h, max_w)` in `mathisfun_sokoban.py` to increase

### Import errors
```bash
# Make sure you're in the correct directory
cd /Users/skr3178/Downloads/math_is_fun

# Make sure conda environment is activated
conda activate sokoban
```

## 📝 Level Format

Levels are stored in `levels_clean.json` as ASCII strings:
```
  ##### 
###   # 
#.@$  # 
### $.# 
#.##$ # 
# # . ##
#$ *$$.#
#   .  #
########
```

Where:
- `#` = Wall
- ` ` = Floor
- `.` = Target
- `$` = Box
- `@` = Player
- `*` = Box on Target
- `+` = Player on Target

## 🙏 Credits

- **Original Game**: Created by Hiroyuki Imabayashi (1981)
- **Level Source**: [MathsIsFun.com](https://www.mathsisfun.com/games/sokoban.html)
- **Environment Author**: Created for Gymnasium/RL training
- **License**: Educational use

## 📚 Additional Resources

- [Sokoban Wikipedia](https://en.wikipedia.org/wiki/Sokoban)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Original Game Site](https://www.mathsisfun.com/games/sokoban.html)

## 🔄 Version History

- **v1.0** - Initial release with all 60 levels
- Enhanced graphics with 3D shading
- Proper window close handling
- Full Gymnasium API compatibility

---

**Enjoy solving puzzles! 🎉**

