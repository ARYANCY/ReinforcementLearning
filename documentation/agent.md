# Agent Overview

In this project, the agent is the wireless transmitter. It observes the jammer state, data queue, and energy queue, then selects the best action to keep communication reliable under interference.

The agent is trained with two reinforcement learning methods: Q-Learning and Deep Q-Network (DQN). Its objective is to maximize long-term throughput while balancing transmission, energy harvesting, ambient backscatter, and rate adaptation.

This learning-based agent allows the system to adapt dynamically in adversarial wireless conditions.
