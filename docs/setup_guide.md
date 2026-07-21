# Setup Guide

> Installation, environment setup, and running instructions for the  
> Ambient Backscatter Anti-Jamming RL framework.

**For a concise run guide (CLI + GUI + manual policy workflow), see [run.md](../run.md).**

---

## 1. Prerequisites

### Required Software
| Requirement | Minimum Version | Notes |
|-------------|-----------------|-------|
| Python | 3.9 or higher | |
| NumPy | 1.22 or higher | Required for all agents |
| TensorFlow | 2.9 or higher | Required only for DQN agent |
| Keras | 2.9 or higher | Included with TensorFlow |
| matplotlib | 3.5 or higher | Optional, for generating plots |
| pytest | Any | Optional, for running tests |

**Note**: Q-Learning runs on **NumPy only** — you don't need to install TensorFlow if you only want to use Q-Learning!

### Quick Install with Requirements File
For easiest installation, use the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

This will install all necessary dependencies (including TensorFlow and matplotlib).

---

## 2. Option A — Conda (Recommended)

```bash
# Create and activate environment
conda create -n ref2_rl python=3.9 -y
conda activate ref2_rl

# Install core dependencies
pip install numpy==1.22.4

# Install TensorFlow (for DQN)
pip install tensorflow==2.9.1 keras==2.9.0

# Install optional plotting
pip install matplotlib
```

---

## 3. Option B — pip (virtualenv)

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Linux/Mac)
source venv/bin/activate

# Install from requirements file
pip install -r requirements.txt
```

---

## 4. Verify Installation

```python
python -c "
import numpy as np
print('NumPy:', np.__version__)
try:
    import tensorflow as tf
    print('TensorFlow:', tf.__version__)
except ImportError:
    print('TensorFlow: not installed (DQN unavailable)')
"
```

---

## 5. Project Structure

```
ref2_rl/
├── parameters.py          ← Hyperparameters (edit here)
├── environment.py         ← MDP environment
├── q_learning_agent.py    ← Q-Learning agent
├── deep_q_agent.py        ← DQN agent
├── train.py               ← Unified training CLI
├── evaluate.py            ← Evaluation + visualisation
├── utils.py               ← Shared utilities
├── requirements.txt       ← Python dependencies
│
├── results/               ← Auto-created on first run
│   ├── q_table.npy
│   ├── q_learning_log.csv
│   ├── dqn_model.keras
│   ├── dqn_log.csv
│   ├── convergence_plot.png
│   └── policy_heatmap.png
│
├── tests/
│   ├── test_environment.py
│   ├── test_q_learning.py
│   └── test_dqn.py
│
└── docs/
    ├── index.md
    ├── architecture.md
    ├── mdp_formulation.md
    ├── algorithms.md
    ├── api_reference.md
    ├── setup_guide.md       ← (this file)
    └── results_guide.md
```

---

## 6. Running the Graphical User Interface (GUI)

The interactive GUI is the easiest way to explore the simulation!

```bash
python gui_app.py
```

### GUI Features:
1. **Real-time Visualization**: Watch the transmitter, receiver, and jammer interact in the simulation canvas
2. **Policy Selection**: Choose from Q-Learning, DQN, Random, or Manual policy modes
3. **Manual Control**: When in Manual mode, click buttons to execute individual actions
4. **Training Controls**: Train Q-Learning and DQN agents directly from the GUI
5. **Q-Table Viewer**: Click "View Q-Table" to see all Q-values in a scrollable table (requires a trained Q-Learning agent)
6. **Simulation Speed**: Adjust the step delay using the slider
7. **Statistics**: View time steps, packets delivered, throughput, and current state
8. **Simulation Log**: Detailed log of all actions and rewards

---

## 7. Running the Experiments (CLI)

### Q-Learning (fast, ~1 minute)

```bash
python train.py --agent q --eval
```

### Deep Q-Network (slower, ~5–15 minutes with GPU)

```bash
python train.py --agent dqn --eval
```

### Both agents with comparison plot

```bash
python train.py --agent both --eval --plot
```

### Evaluate a pre-trained agent

```bash
# Evaluate Q-table
python evaluate.py --agent q --model results/q_table.npy --steps 50000
python evaluate.py --agent q --heatmap   # policy heatmap

# Evaluate DQN model
python evaluate.py --agent dqn --model results/dqn_model.keras --steps 50000
```

---

## 8. Customising Parameters

All parameters are in `parameters.py`. Key things to change:

```python
# Reduce training time for quick tests
T     = 100_000      # Q-Learning steps
T_DQN = 100_000      # DQN steps

# Change jammer behaviour
nu   = 0.3           # More idle jammer
nu_p = [0.8, 0.1, 0.1]  # Jammer mostly uses low power

# Larger queues
d_queue_size = 20
e_queue_size = 20
```

> **Note**: Changing queue sizes will change `num_states`, which invalidates saved Q-tables and DQN models.

---

## 9. Running Tests

The project includes a suite of unit tests to verify all components are working correctly.

### Install pytest
If you haven't already, install pytest:
```bash
pip install pytest
```

### Run all tests
```bash
python -m pytest tests/ -v
```

### Run individual test files
```bash
# Test only the environment
python -m pytest tests/test_environment.py -v

# Test only Q-Learning
python -m pytest tests/test_q_learning.py -v

# Test only DQN
python -m pytest tests/test_dqn.py -v
```

### What the tests check
- `test_environment.py`: Verifies state transitions, reward function, and action feasibility
- `test_q_learning.py`: Verifies Q-table initialization, updates, and selection of feasible actions
- `test_dqn.py`: Verifies replay buffer, network creation, and model saving/loading

---

## 10. Expected Output (Q-Learning)

```
============================================================
 Q-Learning Training
  Steps     : 1,000,000
  α=0.1  γ=0.9  ε-start=1.0
============================================================
  Step   10,000 | ε=0.9048 | Avg reward (last 1000): 0.8234 | Elapsed: 1.2s
  Step   20,000 | ε=0.8187 | Avg reward (last 1000): 1.2451 | Elapsed: 2.4s
  ...
  Step 1,000,000 | ε=0.0100 | Avg reward (last 1000): 2.3871 | Elapsed: 98.1s

Training complete.
  Final avg reward: 2.3871
  Q-table saved → results/q_table.npy
  Training log  → results/q_learning_log.csv
```
