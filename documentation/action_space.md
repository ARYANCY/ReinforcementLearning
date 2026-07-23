# Action Space

The agent can choose from 7 discrete actions. Which actions are valid depends on the current jammer state and queue levels.

| Action | Meaning | When it is allowed |
|---|---|---|
| 0 | Idle | Always |
| 1 | Active transmit (HTT) | Jammer is idle, data is available, and enough energy is stored |
| 2 | Harvest energy | Jammer is active |
| 3 | Backscatter | Jammer is active and data is available |
| 4 | Rate adaptation 0 | Jammer is active and data is available |
| 5 | Rate adaptation 1 | Jammer is active and data is available |
| 6 | Rate adaptation 2 | Jammer is active and data is available |

Infeasible actions are masked before selection so the agent only learns from valid choices. This keeps the policy aligned with the physical constraints of the wireless system.
