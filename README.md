# Reinforcement Learning with Sequential Representation Learning

An experimental **Reinforcement Learning** project focused on learning to play **Atari River Raid** using learned sequential representations and episodic memory.

The project combines ideas from several areas:

- **Atari / River Raid** through [Gymnasium](https://gymnasium.farama.org/)
- A custom **variational sequential autoencoder**, inspired by Disentangled Sequential Autoencoder (DSAE) approaches
- **Neural Episodic Control (NEC)** based on the original NEC paper and its differentiable neural dictionary (DND) memory
- A custom **prioritized and stratified replay memory**
- Experimental studies on replay-memory capacity, sampling strategies, representation learning, and memory management

The main goal is to investigate how **learned sequential representations and episodic memory interact with reinforcement learning**, with the experiments gradually evolving into a custom NEC-based RL architecture.

<video src="best_performance_until_now.mp4" controls width="720"></video>