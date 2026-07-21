# Algorithms

> Derivations and pseudocode for Q-Learning and Deep Q-Network (DQN)

---

## 1. Tabular Q-Learning

### 1.1 Motivation
Q-Learning is a model-free, off-policy reinforcement learning algorithm. For small, discrete MDPs like this one (with only 242 states and 7 actions), tabular Q-Learning is ideal because:
- It is guaranteed to converge to the optimal Q-function under standard conditions
- It is computationally efficient (O(1) per update)
- It requires no deep learning framework

The Q-table stores a value $Q(s,a)$ for every state-action pair, which represents the expected cumulative future reward of taking action $a$ in state $s$ and then following the optimal policy.

### 1.2 Bellman Update
The core of Q-Learning is the Bellman update rule, which iteratively improves the Q-values:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma \max_{a' \in \mathcal{A}(s_{t+1})} Q(s_{t+1}, a') - Q(s_t, a_t) \right]
$$

where:
- $\alpha$: Learning rate (0 < α ≤ 1)
- $\gamma$: Discount factor (0 ≤ γ ≤ 1)
- $r_t$: Reward received at time step $t$
- $\mathcal{A}(s)$: Feasible actions in state $s$

The term $\left[ r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]$ is called the temporal difference (TD) error.

### 1.3 Exploration: ε-Greedy
To balance exploration (trying new actions to discover better policies) and exploitation (choosing the best-known action), we use an ε-greedy strategy with exponential decay:

$$
a_t =
\begin{cases}
\text{random action from } \mathcal{A}(s_t) & \text{with probability } \varepsilon_t \\
\arg\max_{a \in \mathcal{A}(s_t)} Q(s_t, a) & \text{with probability } 1 - \varepsilon_t
\end{cases}
$$

The exploration rate $\varepsilon_t$ decays exponentially over time:

$$
\varepsilon_{t+1} = \max(\varepsilon_{\text{min}}, \varepsilon_t \cdot \delta_\varepsilon)
$$

where:
- $\varepsilon_{\text{min}} = 0.01$: Minimum exploration rate
- $\delta_\varepsilon = 0.9999$: Decay factor per step

### 1.4 Algorithm Pseudocode

```
ALGORITHM: Q-Learning (Tabular)
────────────────────────────────────────────────────────────────────
Input: Number of training steps T, learning rate α, discount factor γ
       initial ε₀, minimum ε_min, decay factor δ_ε

Initialize Q(s,a) = 0 for all s ∈ 𝒮, a ∈ 𝒜(s)
ε ← ε₀
Reset environment to initial state s₀

FOR step t = 1, 2, ..., T:
    𝒜(s_t) ← get_possible_actions(s_t)
    
    IF uniform random(0,1) < ε:
        a_t ← random choice from 𝒜(s_t)  // Explore
    ELSE:
        a_t ← argmax_{a ∈ 𝒜(s_t)} Q(s_t, a)  // Exploit
    
    Execute action a_t, observe reward r_t and next state s_{t+1}
    𝒜(s_{t+1}) ← get_possible_actions(s_{t+1})
    
    // Bellman update
    Q(s_t, a_t) ← Q(s_t, a_t) + α [ r_t + γ max_{a' ∈ 𝒜(s_{t+1})} Q(s_{t+1}, a') - Q(s_t, a_t) ]
    
    // Decay exploration rate
    ε ← max(ε_min, ε · δ_ε)
    
    // Move to next state
    s_t ← s_{t+1}
    
    // (Optional) Log progress every log_interval steps

RETURN Q
────────────────────────────────────────────────────────────────────
```

### 1.5 Hyperparameters

| Hyperparameter | Symbol | Value | Description |
|----------------|--------|-------|-------------|
| Learning rate | α | 0.1 | Step size for Bellman updates |
| Discount factor | γ | 0.9 | Relative importance of future rewards |
| Initial exploration rate | ε₀ | 1.0 | Start with full exploration |
| Minimum exploration rate | ε_min | 0.01 | Always explore a little |
| Exploration decay factor | δ_ε | 0.9999 | Exponential decay per step |
| Training steps | T | 1,000,000 | Total training iterations |

---

## 2. Deep Q-Network (DQN)

### 2.1 Motivation
DQN uses a neural network to approximate the Q-function instead of a table, making it suitable for larger state spaces. Although our state space is small, DQN is included to demonstrate deep RL techniques. DQN introduces two key innovations:
1. **Experience Replay**: Stores past experiences and samples random mini-batches for training to break correlation between consecutive samples
2. **Target Network**: Uses a separate "target" network that is periodically synchronized with the main network to stabilize training

### 2.2 Loss Function
The loss function is the mean squared error between the current Q-network's predictions and the target Q-values:

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( y - Q(s,a; \theta) \right)^2 \right]
$$

where the target $y$ is calculated using the target network with weights $\theta^-$:

$$
y = r + \gamma (1 - \text{done}) \max_{a'} Q(s', a'; \theta^-)
$$

In our case, we don't use a "done" signal since the episode is infinite, so $\text{done}=0$ always.

### 2.3 Network Architecture
The neural network is a simple multi-layer perceptron (MLP):
- **Input layer**: One-hot encoded state vector (242 units)
- **Hidden layer 1**: 64 units with ReLU activation
- **Hidden layer 2**: 64 units with ReLU activation
- **Output layer**: 7 units (one per action) with linear activation

One-hot encoding is used because it treats each state as a distinct entity without introducing artificial ordinal relationships.

### 2.4 Algorithm Pseudocode

```
ALGORITHM: Deep Q-Network (DQN)
────────────────────────────────────────────────────────────────────
Input: Number of training steps T_DQN, learning rate η, discount factor γ
       initial ε₀, minimum ε_min, decay factor δ_ε
       replay buffer size M, mini-batch size B, target update frequency C

Initialize online network Q(s,a; θ) with random weights θ
Initialize target network Q(s,a; θ⁻) with θ⁻ ← θ
Initialize replay buffer 𝒟 with capacity M
ε ← ε₀
Reset environment to initial state s₀

FOR step t = 1, 2, ..., T_DQN:
    φ_t ← one_hot(s_t)  // Convert state to one-hot vector
    𝒜(s_t) ← get_possible_actions(s_t)
    
    IF uniform random(0,1) < ε:
        a_t ← random choice from 𝒜(s_t)  // Explore
    ELSE:
        q_values ← Q(φ_t; θ)
        // Mask infeasible actions by setting Q to -∞
        for a not in 𝒜(s_t):
            q_values[a] ← -∞
        a_t ← argmax(q_values)  // Exploit (masked greedy)
    
    Execute action a_t, observe reward r_t and next state s_{t+1}
    φ_{t+1} ← one_hot(s_{t+1})
    Store experience (φ_t, a_t, r_t, φ_{t+1}, False) in 𝒟
    
    // If buffer has enough experiences, perform a training step
    IF |𝒟| ≥ B:
        Sample mini-batch of B experiences {(φ_i, a_i, r_i, φ'_i, done_i)} from 𝒟
        FOR each experience in mini-batch:
            // Compute target using target network
            max_q' ← max_{a'} Q(φ'_i, a'; θ⁻)
            y_i ← r_i + γ (1 - done_i) max_q'
        
        // Update online network by minimizing MSE(y_i, Q(φ_i, a_i; θ)) via Adam
        θ ← θ - η ∇_θ ℒ(θ)
    
    // Periodically update target network
    IF t mod C == 0:
        θ⁻ ← θ
    
    // Decay exploration rate
    ε ← max(ε_min, ε · δ_ε)
    
    // Move to next state
    s_t ← s_{t+1}
    
    // (Optional) Log progress every log_interval steps

RETURN θ
────────────────────────────────────────────────────────────────────
```

### 2.5 Hyperparameters

| Hyperparameter | Symbol | Value | Description |
|----------------|--------|-------|-------------|
| Adam learning rate | η | 0.001 | Learning rate for the neural network optimizer |
| Discount factor | γ | 0.9 | Relative importance of future rewards |
| Initial exploration rate | ε₀ | 1.0 | Start with full exploration |
| Minimum exploration rate | ε_min | 0.01 | Always explore a little |
| Exploration decay factor | δ_ε | 0.9999 | Exponential decay per step |
| Replay buffer size | M | 2000 | Maximum stored experiences |
| Mini-batch size | B | 32 | Experiences per training step |
| Target update frequency | C | 100 | Steps between target network updates |
| Hidden layer size | - | 64 | Units per hidden layer |
| Number of hidden layers | - | 2 | MLP depth |
| Training steps | T_DQN | 1,000,000 | Total training iterations |

---

## 3. Comparison

| Property | Q-Learning | DQN |
|----------|-----------|-----|
| Q-value representation | Table (242×7 floats) | Neural network weights |
| State space scalability | Small, discrete only | Any (discrete or continuous) |
| Sample efficiency | High (for small state spaces) | Lower (needs large replay buffer) |
| Convergence guarantee | Yes (tabular case) | Approximate only |
| Computational cost per step | O(1) | O(B × network size) |
| Memory usage | Low (small table) | Higher (two networks + buffer) |
| Hyperparameter sensitivity | Low | Moderate |

---

## 4. Expected Anti-Jamming Strategy

After convergence, the agents typically learn the following intuitive policy:

| State Condition | Optimal Action |
|-----------------|---------------|
| Jammer idle ($j=0$), $d>0$, $e \geq e_t$ | Active Transmit (1) — maximize throughput when jammer is quiet |
| Jammer active ($j=1$), high power | Harvest Energy (2) — save up energy for future active transmit |
| Jammer active ($j=1$), low power | Backscatter (3) — opportunistically use jammer signal to transmit |
| Jammer active ($j=1$), medium power | Rate Adaptation (4-6) — use best possible rate |
| Data queue empty ($d=0$) | Idle or Harvest — no data to send |
| Energy queue empty ($e=0$) | Harvest or Backscatter — no energy for active transmit |

This demonstrates that the agent learns to view the jammer not just as an obstacle, but also as a resource for energy harvesting and backscatter communication!
