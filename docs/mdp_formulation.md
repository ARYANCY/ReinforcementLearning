# MDP Formulation

> Full mathematical specification of the Markov Decision Process  
> underlying the Ambient Backscatter Anti-Jamming framework.

---

## 1. Background

This problem is modeled as a discrete-time, discounted Markov Decision Process (MDP) defined as:

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)
$$

where:
- $\mathcal{S}$: State space (all possible states the system can be in)
- $\mathcal{A}$: Action space (all possible actions the agent can take)
- $\mathcal{P}$: Transition probability function ($\mathcal{P}(s'|s,a)$: probability of transitioning to $s'$ from $s$ by taking action $a$)
- $\mathcal{R}$: Reward function ($\mathcal{R}(s,a,s')$: expected reward received after transitioning from $s$ to $s'$ via action $a$)
- $\gamma$: Discount factor (0 ≤ γ ≤ 1), which determines the importance of future rewards relative to immediate rewards

In this framework, the transmitter acts as the RL agent, and the jammer, queues, and data source form the environment.

---

## 2. State Space $\mathcal{S}$

Each state is a tuple consisting of three discrete components:

$$
s = (j, d, e)
$$

| Component | Symbol | Domain | Description |
|-----------|--------|--------|-------------|
| Jammer state | $j$ | $\{0, 1\}$ | 0 = jammer idle, 1 = jammer active |
| Data queue occupancy | $d$ | $\{0, 1, ..., D\}$ | Number of packets in the data queue (D=10) |
| Energy queue occupancy | $e$ | $\{0, 1, ..., E\}$ | Number of energy units in the energy queue (E=10) |

### Total Number of States
The total number of possible states is:

$$
|\mathcal{S}| = 2 \times (D+1) \times (E+1) = 2 \times 11 \times 11 = 242
$$

### State Encoding
Each state tuple $(j, d, e)$ is encoded into a single integer index for efficient lookup:

$$
\text{idx}(j, d, e) = j \cdot (D+1) \cdot (E+1) + d \cdot (E+1) + e
$$

The decoding function to retrieve $(j, d, e)$ from an index is:

$$
j = \left\lfloor \frac{\text{idx}}{(D+1)(E+1)} \right\rfloor
$$
$$
\text{remaining} = \text{idx} \mod (D+1)(E+1)
$$
$$
d = \left\lfloor \frac{\text{remaining}}{(E+1)} \right\rfloor
$$
$$
e = \text{remaining} \mod (E+1)
$$

---

## 3. Action Space $\mathcal{A}$

The action space consists of 7 possible actions:

| Index | Action Name | Feasibility Conditions | Description |
|-------|-------------|------------------------|-------------|
| 0 | Idle | Always feasible | Do nothing; no packets transmitted, no energy harvested |
| 1 | Active Transmit (HTT) | $j=0$, $d>0$, $e \geq e_t$ | Transmit $d_t$ packets using harvested energy; only possible when jammer is idle |
| 2 | Harvest Energy | $j=1$ | Harvest energy from jammer's signal; only possible when jammer is active |
| 3 | Backscatter | $j=1$, $d>0$ | Backscatter jammer's signal to transmit data; only possible when jammer is active |
| 4 | Rate Adaptation 0 | $j=1$, $d>0$ | Transmit with rate adaptation level 0 |
| 5 | Rate Adaptation 1 | $j=1$, $d>0$ | Transmit with rate adaptation level 1 |
| 6 | Rate Adaptation 2 | $j=1$, $d>0$ | Transmit with rate adaptation level 2 |

### Action Masking
Before any action is selected, infeasible actions (those that don't meet the conditions above) are masked (their Q-values are set to $-\infty$) to prevent the agent from choosing them.

---

## 4. Jammer Model

The jammer operates with two types of randomness:

### 4.1 Jammer State Transition
The jammer switches between idle and active states according to a 2-state Markov chain with transition probabilities:

$$
\Pr(j_{t+1} = 0) = \nu = 0.1
$$
$$
\Pr(j_{t+1} = 1) = 1 - \nu = 0.9
$$

Note that the jammer state transition is independent of the agent's actions.

### 4.2 Jammer Power Level
When active ($j=1$), the jammer uses one of three discrete power levels, chosen randomly according to the distribution:

$$
\boldsymbol{\nu}_p = [0.6, 0.2, 0.2]
$$

This means:
- Power level 0: 60% probability
- Power level 1: 20% probability
- Power level 2: 20% probability

The transmitter does not observe the exact power level; it only observes whether the jammer is idle or active.

---

## 5. Reward Function $\mathcal{R}(s, a)$

The reward function is the number of packets successfully delivered to the receiver in one time slot.

### Action 0: Idle
$$
\mathcal{R}(s, 0) = 0
$$

### Action 1: Active Transmit (HTT)
If jammer is idle, $d>0$, and $e \geq e_t$:
$$
\mathcal{R}(s, 1) = \min(d_t, d)
$$
Otherwise:
$$
\mathcal{R}(s, 1) = 0
$$

where $d_t=4$ (packets per slot) and $e_t=1$ (energy per packet).

### Action 2: Harvest Energy
$$
\mathcal{R}(s, 2) = 0
$$
Harvesting energy gives no immediate reward, but increases the energy queue for future use.

### Action 3: Backscatter
If jammer is active and $d>0$:
$$
\mathcal{R}(s, 3) = \min(d_{bj}[p], d)
$$
Otherwise:
$$
\mathcal{R}(s, 3) = 0
$$

where $d_{bj} = [1,2,3]$ (packets delivered per power level) and $p$ is the jammer's power level.

### Actions 4-6: Rate Adaptation
If jammer is active and $d>0$:
$$
\mathcal{R}(s, 4+m) = \min(d_{ta}[m], d)
$$
Otherwise:
$$
\mathcal{R}(s, 4+m) = 0
$$

where $d_{ta} = [2,1,0]$ (packets delivered per rate adaptation level) and $m \in \{0,1,2\}$.

---

## 6. Transition Dynamics $\mathcal{P}$

The state transition from $s=(j,d,e)$ to $s'=(j',d',e')$ is determined by three independent components:

### 6.1 Jammer State Transition
$$
j' \sim \text{Bernoulli}(1 - \nu)
$$
As described in Section 4.1, this transition is independent of the agent's action.

### 6.2 Queue Updates
The data and energy queue updates depend on the chosen action $a$:

#### Action 0: Idle
$$
d' = \min(d + A_t, D)
$$
$$
e' = e
$$

#### Action 1: Active Transmit
If feasible:
$$
d' = \min(d - \min(d_t, d) + A_t, D)
$$
$$
e' = e - \min(\min(d_t, d) \cdot e_t, e)
$$
Otherwise:
$$
d' = \min(d + A_t, D)
$$
$$
e' = e
$$

#### Action 2: Harvest Energy
If feasible:
$$
d' = \min(d + A_t, D)
$$
$$
e' = \min(e + e_{hj}[p], E)
$$
Otherwise:
$$
d' = \min(d + A_t, D)
$$
$$
e' = e
$$

where $e_{hj} = [1,2,3]$ (energy harvested per power level).

#### Action 3: Backscatter
If feasible:
$$
d' = \min(d - \min(d_{bj}[p], d) + A_t, D)
$$
$$
e' = e
$$
Otherwise:
$$
d' = \min(d + A_t, D)
$$
$$
e' = e
$$

#### Actions 4-6: Rate Adaptation
If feasible:
$$
d' = \min(d - \min(d_{ta}[m], d) + A_t, D)
$$
$$
e' = e
$$
Otherwise:
$$
d' = \min(d + A_t, D)
$$
$$
e' = e
$$

### 6.3 Poisson Arrivals
The number of new packets arriving at the data queue in each time slot is a Poisson random variable with arrival rate $\lambda=3$:

$$
A_t \sim \text{Poisson}(\lambda = 3)
$$

---

## 7. Optimization Objective

The agent's goal is to maximize the average long-run throughput:

$$
\max_\pi R(\pi) = \lim_{T \to \infty} \frac{1}{T} \sum_{t=1}^{T} \mathbb{E}_\pi\left[\mathcal{R}(s_t, a_t)\right]
$$

For practical purposes, we use a discounted reward formulation as a surrogate (which often leads to near-optimal average-reward policies):

$$
\max_\pi V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t \mathcal{R}(s_t, a_t) \mid s_0 = s\right]
$$

### Bellman Optimality Equation
The optimal Q-value function $Q^*$ satisfies the Bellman optimality equation:

$$
Q^*(s, a) = \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s'|s,a) \max_{a' \in \mathcal{A}(s')} Q^*(s', a')
$$

where $\mathcal{A}(s)$ is the set of feasible actions in state $s$.

---

## 8. Fixed-Point Summary Table

| Symbol | Value | Description |
|--------|-------|-------------|
| $\nu$ | 0.1 | Jammer idle probability |
| $\lambda$ | 3 | Poisson packet arrival rate |
| $D$ | 10 | Data queue capacity |
| $E$ | 10 | Energy queue capacity |
| $d_t$ | 4 | Active transmit packets per slot |
| $e_t$ | 1 | Energy per packet for active transmit |
| $d_{bj}$ | $[1, 2, 3]$ | Backscatter packets per power level |
| $e_{hj}$ | $[1, 2, 3]$ | Harvested energy per power level |
| $d_{ta}$ | $[2, 1, 0]$ | Rate adaptation packets per level |
| $b^\dagger$ | 3 | Fixed backscatter rate |
| $\gamma$ | 0.9 | Discount factor |
| $\alpha$ | 0.1 | Q-Learning learning rate |
