# State Space

The state is represented by three values: jammer state, data queue level, and energy queue level. Together, these describe the transmitter's current operating condition.

The jammer has two states, while both queues range from 0 to 10, giving a total of 242 possible states. This makes the environment small enough for tabular learning, while still rich enough to model realistic decisions.

The agent uses this state to decide whether to transmit, harvest, backscatter, or wait.
