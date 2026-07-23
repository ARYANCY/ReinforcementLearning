# Introduction

This project uses reinforcement learning to improve wireless communication under jamming attacks. A transmitter learns which action to take from the jammer state, data queue, and energy queue.

We model the problem as a Markov Decision Process and solve it with Q-Learning and Deep Q-Networks (DQN). The objective is to maximize throughput while balancing transmission, energy harvesting, ambient backscatter, and rate adaptation.
