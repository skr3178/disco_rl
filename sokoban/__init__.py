# __init__.py
"""
MathIsFun Sokoban - A Gymnasium Environment

A complete Gymnasium-compatible environment featuring all 60 Sokoban puzzle 
levels from MathsIsFun.com. Perfect for reinforcement learning research and training.
"""

__version__ = "1.0.0"

from gymnasium.envs.registration import register

# Try both import methods for compatibility
try:
    from mathisfun_sokoban import MathIsFunSokoban
except ImportError:
    try:
        from .mathisfun_sokoban import MathIsFunSokoban
    except ImportError:
        # If installed as package, try absolute import
        import sys
        import pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).parent))
        from mathisfun_sokoban import MathIsFunSokoban

register(
    id="MathIsFunSokoban-v0",
    entry_point="mathisfun_sokoban:MathIsFunSokoban",
    max_episode_steps=300,
)

__all__ = ["MathIsFunSokoban", "__version__"]

