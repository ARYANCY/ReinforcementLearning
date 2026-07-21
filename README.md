# Ambient Backscatter Anti-Jamming via Reinforcement Learning

> **Reference**: N. Van Huynh, D. N. Nguyen, D. T. Hoang, E. Dutkiewicz, and M. Mueck,  
> "Ambient backscatter: A novel method to defend jamming attacks for wireless networks,"  
> *IEEE Wireless Communications Letters*, vol. 9, no. 2, pp. 175–178, 2019.

---

## Overview

This repository implements a **reinforcement learning-based anti-jamming framework** where a wireless transmitter learns to defend against jamming attacks by:

- **Ambient Backscatter** — reflecting/modulating jamming signals to transmit data
- **Energy Harvesting (HTT)** — converting jamming RF energy into transmit power
- **Rate Adaptation** — adjusting transmission rate to jammer power level

The system is modelled as a **Markov Decision Process (MDP)** and solved using:
- **Q-Learning** — tabular, exact, fast to converge
- **Deep Q-Network (DQN)** — neural network approximator, scalable

---

## Project Structure

```
ref2_rl/
├── parameters.py          ← All hyperparameters
├── environment.py         ← MDP environment
├── q_learning_agent.py    ← Tabular Q-Learning
├── deep_q_agent.py        ← Deep Q-Network (DQN)
├── train.py               ← CLI training entry point
├── evaluate.py            ← Evaluation + policy heatmap
├── utils.py               ← Shared utilities
├── gui_app.py             ← Interactive RL visualizer
├── run.md                 ← Complete run guide (CLI + GUI + manual mode)
├── requirements.txt       ← Python dependencies
│
├── results/               ← Generated: models, logs, plots
│
├── tests/
│   ├── test_environment.py
│   ├── test_q_learning.py
│   └── test_dqn.py
│
└── docs/
    ├── index.md            ← Documentation home
    ├── architecture.md     ← System architecture + DFD diagrams
    ├── mdp_formulation.md  ← MDP theory with LaTeX equations
    ├── algorithms.md       ← Q-Learning & DQN derivations + pseudocode
    ├── api_reference.md    ← Module-level API reference
    ├── setup_guide.md      ← Installation & running
    └── results_guide.md    ← Interpreting outputs
```

---

## Quick Start

See **[run.md](run.md)** for the complete running guide (CLI, GUI, manual policy workflow, troubleshooting).

### 1. Install dependencies

```bash
# Core (Q-Learning only)
pip install numpy matplotlib pytest

# Add TensorFlow for DQN
pip install tensorflow==2.9.1
```

### 2. Train

```bash
# Q-Learning (fast, ~1 min)
python train.py --agent q --eval

# DQN (~5-15 min with GPU)
python train.py --agent dqn --eval

# Both + comparison plot
python train.py --agent both --eval --plot
```

### 3. Evaluate saved model

```bash
python evaluate.py --agent q --heatmap
```

### 4. Run tests

```bash
python -m pytest tests/ -v
```

---

## MDP Summary

| Component | Description |
|-----------|-------------|
| **State** | $(j, d, e)$: jammer state, data queue, energy queue |
| **Actions** | Idle / Active-TX / Harvest / Backscatter / Rate-Adapt (×3) |
| **Reward** | Packets successfully delivered per slot |
| **Objective** | Maximise long-run average throughput |
| **State space** | $2 \times 11 \times 11 = 242$ states |
| **Action space** | 7 actions |

---

## Key Results (Expected)

| Metric | Q-Learning | DQN |
|--------|-----------|-----|
| Final avg reward | ~2.3–2.5 packets/slot | ~2.2–2.4 packets/slot |
| Convergence | ~200k steps | ~400k steps |
| Training time | ~1–2 min | ~5–15 min |

---

## Graphical User Interface (GUI)

We provide an interactive GUI for visualizing and controlling the simulation!

### Features:
- Real-time visualization of:
  - Transmitter, receiver, and jammer state
  - Data and energy queue levels
  - Jammer activity (idle/active)
- Policy selector (Q-Learning, DQN, Random, Manual)
- Manual action controls for step-by-step testing
- **Q-Table Viewer**: View the entire Q-table in a scrollable table
- Training controls for Q-Learning and DQN
- Simulation log and statistics

### Run the GUI:
```bash
python gui_app.py
```

---

## Documentation

All detailed documentation is available in the [`docs/`](docs/index.md) directory:

| File | Description |
|------|-------------|
| [Documentation Home](docs/index.md) | Navigation and overview of all docs |
| [Architecture](docs/architecture.md) | System design, data flow diagrams, component breakdown |
| [MDP Formulation](docs/mdp_formulation.md) | Full mathematical specification |
| [Algorithms](docs/algorithms.md) | Q-Learning & DQN derivations and pseudocode |
| [API Reference](docs/api_reference.md) | Complete module-level API docs |
| [Setup Guide](docs/setup_guide.md) | Installation, configuration, and running |
| [Results Guide](docs/results_guide.md) | Interpreting outputs and metrics |

---

## Citation

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
