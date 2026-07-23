# Environment Overview

The environment models a wireless anti-jamming system in discrete time slots. At each step, the transmitter observes the jammer state, data queue, and energy queue, then chooses an action that affects future communication performance.

The environment includes three key elements: a jammer that switches between idle and active states, a data source that feeds packets into the queue, and an energy queue that stores harvested power for later transmission.

This setup gives the RL agent a realistic control problem: use the jammer as both a threat and a resource, while maximizing long-term packet delivery under physical constraints.
