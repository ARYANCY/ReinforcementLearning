# Ambient Backscatter Anti-Jamming via Reinforcement Learning

> **Reference**: N. Van Huynh, D. N. Nguyen, D. T. Hoang, E. Dutkiewicz, and M. Mueck,  
> "Ambient backscatter: A novel method to defend jamming attacks for wireless networks,"  
> *IEEE Wireless Communications Letters*, vol. 9, no. 2, pp. 175–178, 2019.

---

## 📌 Project Overview

This repository implements a **reinforcement learning-based anti-jamming framework** for low-power IoT and wireless networks operating under interference and energy constraints. Rather than treating the jammer solely as a destructive interference source, the wireless transmitter dynamically adapts its operational mode to turn jamming energy into a usable resource:

- **Active Transmit (Harvest-Then-Transmit - HTT)**: Transmits data packets directly when the jammer is idle, consuming stored battery energy.
- **Energy Harvesting (EH)**: Harvests RF energy from active jamming signals to charge the energy queue.
- **Ambient Backscatter (AB)**: Modulates and reflects the incoming jamming signal to transmit data without expending local battery power.
- **Rate Adaptation (RA)**: Adjusts transmission rate levels based on observed jamming power.

The decision-making process is modeled as a **Markov Decision Process (MDP)** and solved using two reinforcement learning approaches:
1. **Tabular Q-Learning**: Exact model-free learning, fast convergence for discrete state spaces.
2. **Deep Q-Network (DQN)**: Neural network function approximator with experience replay and target network synchronization.

---

## 🏗️ System Architecture & Components

The framework comprises three primary physical entities interacting with a reinforcement learning control layer:

1. **Transmitter (RL Agent)**: The decision-making node. It observes the environment state $(j, d, e)$, selects actions dynamically, and balances energy harvesting, backscattering, rate adaptation, and active transmission.
2. **Jammer**: The source of interference. It transitions between **Idle** and **Active** states according to a Markov process and emits RF signals at different power levels when active.
3. **Receiver**: The destination node receiving packets delivered via active transmission, backscatter, or rate adaptation modes.
4. **Queues**:
   - **Data Queue ($d \in [0, 10]$)**: Buffers incoming Poisson packet arrivals.
   - **Energy Queue ($e \in [0, 10]$)**: Stores harvested RF energy units.

```
                  +-------------------------+
                  |     Jammer State        |
                  |  (Idle / Active Power)  |
                  +------------+------------+
                               |
            Interference / RF | Energy
                               v
+------------------+     +-----+-----+     +------------------+
|   Data Arrivals  | --> | Transmitter | --> |     Receiver     |
| (Poisson Stream) |     |  (RL Agent) |     | (Data Destination|
+------------------+     +-----+-----+     +------------------+
                               |
                        +------+------+
                        |  Queues:    |
                        |  - Data (d) |
                        |  - Energy(e)|
                        +-------------+
```

---

## 📐 MDP Formulation

| Component | Description | Details |
|---|---|---|
| **State Space** | $s = (j, d, e)$ | $j \in \{0, 1\}$ (Jammer idle/active), $d \in [0, 10]$ (Data queue), $e \in [0, 10]$ (Energy queue). Total: **242 states**. |
| **Action Space** | 7 Discrete Actions | Action masking filters infeasible choices per state. |
| **Reward Function** | $R(s, a)$ | Number of packets successfully delivered to receiver in the current time slot. |
| **Objective** | $\max \mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^t R_t \right]$ | Maximize long-term average packet throughput under battery & queue stability. |

### Action Space Summary

| Action ID | Action Name | Prerequisite / Condition | Effect / Outcome |
|:---:|---|---|---|
| **0** | Idle | Always allowed | No transmission or harvesting |
| **1** | Active Transmit | Jammer Idle, $d > 0$, $e \ge e_t$ | Transmits packets using stored energy |
| **2** | Energy Harvest | Jammer Active | Converts jamming RF signal into energy units |
| **3** | Ambient Backscatter | Jammer Active, $d > 0$ | Reflects jamming signal to transmit data |
| **4** | Rate Adaptation 0 | Jammer Active, $d > 0$ | Transmits at rate adaptation level 0 |
| **5** | Rate Adaptation 1 | Jammer Active, $d > 0$ | Transmits at rate adaptation level 1 |
| **6** | Rate Adaptation 2 | Jammer Active, $d > 0$ | Transmits at rate adaptation level 2 |

---

## 🤖 Reinforcement Learning Algorithms

- **Q-Learning**:
  - Updates $Q(s,a) \leftarrow Q(s,a) + \alpha \left[ R + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]$.
  - Fast convergence (~200,000 steps), minimal memory footprint.
- **Deep Q-Network (DQN)**:
  - Uses an MLP Q-network with hidden layers `[64, 64]`.
  - Employs **Experience Replay Buffer** and **Target Network Synchronization** to break temporal correlations and ensure stable gradient updates.

---

## 📁 Repository Structure

```
ref2_rl/
├── parameters.py            # Centralized system & hyperparameter settings
├── environment.py           # MDP Environment implementation
├── q_learning_agent.py      # Tabular Q-Learning agent
├── deep_q_agent.py          # Deep Q-Network (DQN) agent (Keras 3 / TF 2.x)
├── train.py                 # CLI training script
├── evaluate.py              # Evaluation suite & policy heatmap generator
├── utils.py                 # Common helper functions & plotters
├── gui_app.py               # Interactive Light-Theme GUI Visualizer (Tkinter)
├── run.md                   # Command-line & GUI execution guide
├── requirements.txt         # Project dependencies
│
├── documentation/           # Modular project documentation suite
│   ├── introduction.md      # Background and problem context
│   ├── project_overview.md  # High-level goals & methodology
│   ├── system_architecture.md# Wireless & learning layer architecture
│   ├── transmitter.md       # Decision-making node details
│   ├── jammer.md            # Jammer transition & power model
│   ├── environment.md       # MDP environment mechanics
│   ├── state_space.md       # State representation & indexing
│   ├── action_space.md       # Action definitions & feasibility masking
│   ├── reward_function.md   # Packet throughput reward design
│   ├── working.md           # Step-by-step simulation loop
│   ├── agent.md             # RL agent structure & learning loops
│   ├── Q_Learning.md        # Tabular Q-learning formulation
│   ├── Deep_Q_Network.md    # DQN architecture & replay mechanics
│   ├── results.md           # Performance analysis & takeaways
│   └── references.md        # Academic papers & references
│
├── results/                 # Output artifacts (logs, models, heatmap plots)
└── tests/                   # Pytest automated test suite
    ├── test_environment.py
    ├── test_q_learning.py
    └── test_dqn.py
```

---

## 🚀 Quick Start

For detailed execution workflows, refer to **[run.md](run.md)**.

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ref2_rl.git
cd ref2_rl

# Install dependencies
pip install -r requirements.txt
```

### 2. Train RL Agents

```bash
# Train tabular Q-Learning agent (~1 min)
python train.py --agent q --eval

# Train Deep Q-Network agent (~5-15 min)
python train.py --agent dqn --eval

# Train both agents and generate comparative reward plot
python train.py --agent both --eval --plot
```

### 3. Evaluate Policies & Heatmaps

```bash
python evaluate.py --agent q --heatmap
```

### 4. Run Automated Test Suite

```bash
python -m pytest tests/ -v
```

---

## 💻 Interactive GUI Visualizer

Launch the light-themed Tkinter dashboard for real-time visual demonstration:

```bash
python gui_app.py
```

### Key GUI Features:
- **Real-Time Canvas Animation**: Displays transmitter mast, satellite receiver dish, radiating jammer waves, and non-blocking packet pulses.
- **Queue Gauges**: Live data buffer and energy buffer progress meters.
- **Policy Switcher**: Switch between **Q-Learning**, **DQN**, **Random**, and **Manual** policies on the fly.
- **Manual Control Pad**: Step through discrete actions manually with dynamic feasibility masking.
- **In-App Training**: Trigger Q-Learning or DQN background training directly from the UI.
- **Live Telemetry & Logs**: Real-time console log tracking packet throughput, action history, and status updates.

---

## 📖 Comprehensive Documentation Suite

Full module documentation is organized inside the [`documentation/`](documentation/) directory:

| Document | Topic / Content |
|---|---|
| [Introduction](documentation/introduction.md) | Problem background and anti-jamming context |
| [Project Overview](documentation/project_overview.md) | High-level system objectives & summary |
| [System Architecture](documentation/system_architecture.md) | Wireless layer & learning integration |
| [Transmitter Node](documentation/transmitter.md) | Decision-making node capabilities |
| [Jammer Model](documentation/jammer.md) | Jammer state transitions & power levels |
| [Environment](documentation/environment.md) | Discrete time-slot MDP simulation |
| [State Space](documentation/state_space.md) | $(j, d, e)$ state representation (242 states) |
| [Action Space](documentation/action_space.md) | 7 discrete actions & feasibility masking rules |
| [Reward Function](documentation/reward_function.md) | Packet delivery reward formulation |
| [Working Process](documentation/working.md) | Time-slot execution loop |
| [Agent Overview](documentation/agent.md) | RL Agent structure & role |
| [Q-Learning](documentation/Q_Learning.md) | Tabular Q-learning update equations |
| [Deep Q-Network](documentation/Deep_Q_Network.md) | DQN neural architecture & experience replay |
| [Results Overview](documentation/results.md) | Performance evaluation & convergence gains |
| [References](documentation/references.md) | Academic citations & paper details |

---

## 📊 Expected Performance & Results

| Metric | Tabular Q-Learning | Deep Q-Network (DQN) |
|---|:---:|:---:|
| **Convergence Speed** | ~200,000 steps | ~400,000 steps |
| **Final Avg Throughput** | ~2.3 – 2.5 pkts/slot | ~2.2 – 2.4 pkts/slot |
| **Training Time** | ~1 minute | ~5–15 minutes |
| **Memory Footprint** | Low ($242 \times 7$ table) | Moderate (TensorFlow Model) |

---

## 📚 Citation

```bibtex
@article{huynh2019ambient,
  author  = {Van Huynh, Nguyen and Nguyen, Dinh Ngu and Hoang, Diep T.
             and Dutkiewicz, Eryk and Mueck, Markus},
  title   = {Ambient backscatter: A novel method to defend jamming attacks
             for wireless networks},
  journal = {IEEE Wireless Communications Letters},
  volume  = {9},
  number  = {2},
  pages   = {175--178},
  year    = {2019}
}
```
