# Transmitter Overview

The transmitter is the decision-making node in this project. It observes the jammer state, data queue, and energy queue, then selects the action most likely to improve future communication.

It can transmit actively, harvest energy, backscatter, or adapt its rate depending on the jammer's behavior. These capabilities let the transmitter use the jammer as both an interference source and a resource.

In the RL framework, the transmitter is the agent being trained to maximize long-term throughput.
