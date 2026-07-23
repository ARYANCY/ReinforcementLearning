# System Architecture

The system has three physical components: a transmitter, a receiver, and a jammer. A data source feeds packets into the transmitter, while the jammer alternates between idle and active states.

The reinforcement learning agent sits on top of this communication process and selects actions based on the observed state. Its job is to maximize successful packet delivery while respecting queue and energy constraints.

This architecture combines the wireless layer with the learning layer, allowing the transmitter to adapt to interference in real time.
