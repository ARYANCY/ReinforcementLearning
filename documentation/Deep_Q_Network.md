# Deep Q Network

Deep Q Network (DQN) is the deep reinforcement learning method used in this project to learn anti-jamming policies for wireless communication. Instead of storing Q-values in a table, it uses a neural network to approximate the best action for each state.

The agent observes the jammer state, data queue, and energy queue, then learns when to transmit, harvest energy, backscatter, or adapt the rate. A target network and experience replay are used to make training more stable.

In this project, DQN provides a scalable alternative to tabular Q-Learning while aiming to maximize long-term throughput under adversarial jamming.
