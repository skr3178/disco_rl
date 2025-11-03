#!/bin/bash
# Quick training script

echo "🚀 Installing dependencies..."
pip install -e .
pip install "stable-baselines3[extra]" torch

echo ""
echo "🎮 Starting training with rendering..."
echo "A pygame window will open showing the agent learning."
echo ""

python example_stable_baselines3.py

