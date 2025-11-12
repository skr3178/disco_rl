# To do next

- Create a symbolic link /training is faster with symbolic link
- 
- Train with the disco-rl weights
- 


⚙️ How It Works

The system has two learning levels:

Agent level (inner loop):
Each agent interacts with its environment and updates itself using a current RL rule.

Meta level (outer loop):
The system learns the RL rule itself — it updates the parameters of the “RL rule network” (called the meta-network) based on how much reward the agents achieve.

So, the RL update rule (normally hard-coded by humans) is instead a neural network learned by meta-gradient optimization.

🧩 Agent Components

Each agent outputs:

A policy (π)

State-based predictions y(s)

Action-based predictions z(s,a)

Optional predefined predictions (value function q(s,a), auxiliary policy p(s,a))

These predictions are flexible — they can represent anything useful the meta-network discovers (e.g., analogs of values, advantages, or new internal quantities).

🧮 Meta-Network (the discovered RL rule)

The meta-network:

Takes trajectories (sequences of predictions, policies, rewards)

Uses an LSTM to process future and past steps

Outputs targets for all agent outputs (policy, y, z, etc.)

The agent then updates itself to minimize the KL divergence between its outputs and these targets.

🔁 Meta-Optimization

The goal is to maximize the total reward of all agents using the discovered rule.

Mathematically:

∇_η J(η) ≈ E_{E,θ} [∇_η θ ∇_θ J(θ)]

where:

- J(η): meta-objective (total reward)
- η: meta-network parameters
- θ: agent parameters

They use meta-gradients to update η — i.e., backpropagation through the agent learning process.
