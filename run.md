# How to Run — Ambient Backscatter Anti-Jamming RL

Complete guide for installing, training, evaluating, visualizing, and running the interactive GUI for this project.

---

## Prerequisites

| Requirement | Version | Required For |
|-------------|---------|--------------|
| Python | 3.9+ | Everything |
| NumPy | 1.22+ | Q-Learning, environment, GUI |
| TensorFlow | 2.9+ | DQN training & GUI DQN policy |
| matplotlib | 3.5+ | Plots, heatmaps (optional but recommended) |
| pytest | any | Unit tests (optional) |

### Install (recommended)

```bash
# From project root
pip install -r requirements.txt
```

**Q-Learning only** (no TensorFlow):

```bash
pip install numpy matplotlib pytest
```

### Verify installation

```bash
python -c "import numpy; print('NumPy OK')"
python -c "import tensorflow as tf; print('TensorFlow', tf.__version__)"
```

---

## Project Layout

```
ref2_rl/
├── parameters.py         # All hyperparameters (edit here)
├── environment.py        # MDP: states, actions, rewards
├── q_learning_agent.py   # Tabular Q-Learning agent
├── deep_q_agent.py       # Deep Q-Network (DQN) agent
├── train.py              # CLI training entry point
├── evaluate.py           # Evaluate saved models + heatmap
├── generate_plots.py     # Generate all 5 result plots
├── gui_app.py            # Interactive visualizer (recommended for demos)
├── utils.py              # Shared helpers
├── results/              # Auto-created with subdirectories
│   ├── models/           # Saved Q-table and DQN model
│   ├── logs/             # Training CSV logs
│   └── plots/            # Generated PNG plots
├── documentation/        # Project documentation
└── tests/                # Unit tests (45 tests)
```

---

## Quick Start (5 steps)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train Q-Learning for 10,000 steps (fast)
python train.py --agent q --steps 10000 --eval

# 3. Generate all result plots
python generate_plots.py

# 4. (Optional) Train DQN
python train.py --agent dqn --steps 10000 --eval

# 5. Launch interactive GUI
python gui_app.py
```

---

## CLI Training (`train.py`)

### Q-Learning (fast, tabular)

Trains a 242×7 Q-table. Best for quick experiments and the Q-Table viewer in the GUI.

```bash
python train.py --agent q --eval --plot
```

**Outputs:**
- `results/models/q_table.npy` — trained Q-table
- `results/logs/q_learning_log.csv` — convergence log with step, reward, avg_reward, cumulative_reward

### Deep Q-Network (DQN)

Neural-network approximation. Slower but scales to larger state spaces.

```bash
python train.py --agent dqn --eval --plot
```

**Outputs:**
- `results/models/dqn_model.keras` — saved Keras model
- `results/logs/dqn_log.csv` — convergence log with step, reward, avg_reward, cumulative_reward

### Train both + comparison plots

```bash
python train.py --agent both --eval --plot
```

### CLI options for `train.py`

| Flag | Description | Default |
|------|-------------|---------|
| `--agent {q,dqn,both}` | Which agent to train | `q` |
| `--eval` | Run greedy evaluation after training | off |
| `--seed INT` | Random seed | `42` |
| `--plot` | Generate all 5 plots after training (runs `generate_plots.py`) | off |
| `--steps INT` | Custom number of training steps | None (uses value from `parameters.py`) |

---

## Evaluation (`evaluate.py`)

Evaluate a pre-trained agent without retraining:

```bash
# Q-Learning (50k steps)
python evaluate.py --agent q --steps 50000 --heatmap

# DQN (50k steps)
python evaluate.py --agent dqn --steps 50000

# Custom model path
python evaluate.py --agent q --model path/to/q_table.npy

# Generate all plots after evaluation
python evaluate.py --agent q --plot
```

**Heatmap output:** `results/plots/04_q_table_heatmap.png`

### CLI options for `evaluate.py`

| Flag | Description | Default |
|------|-------------|---------|
| `--agent {q,dqn}` | Which agent to evaluate (required) | None |
| `--model PATH` | Custom path to saved model | None (uses `results/models/` path) |
| `--steps INT` | Number of evaluation steps | 50000 |
| `--plot` | Generate all 5 result plots | off |
| `--heatmap` | Generate Q-table heatmap (Q-Learning only) | off |

---

## Generate Plots (`generate_plots.py`)

Generate all 5 result plots from existing training logs:

```bash
python generate_plots.py
```

This script will:
1. Look for CSV logs in `results/logs/` (with fallback to `results/`)
2. Look for Q-table in `results/models/` (with fallback to `results/`)
3. Generate all 5 plots in `results/plots/`

The 5 plots are:
1. `01_reward_vs_training_steps.png`
2. `02_average_throughput_vs_training_steps.png`
3. `03_cumulative_reward_vs_training_steps.png`
4. `04_q_table_heatmap.png`
5. `05_action_selection_distribution.png`

---

## Interactive GUI (`gui_app.py`)

The GUI is the best way to **see** the anti-jamming system in action and to **manually test** individual actions.

```bash
python gui_app.py
```

### GUI overview

| Panel | Purpose |
|-------|---------|
| **Canvas** | Transmitter, receiver, jammer, data/energy queues |
| **Side panel** | Jammer status, queue meters, last action, speed slider |
| **Stats bar** | Timesteps, packets delivered, throughput, current state |
| **Simulation log** | Step-by-step action/reward history |
| **Control bar** | Policy selector, Run/Step/Reset, training buttons |

### Policy modes

| Policy | Behaviour |
|--------|-----------|
| **Q-Learning** | Greedy action from saved Q-table (`results/models/q_table.npy`) |
| **Deep Q-Network (DQN)** | Greedy action from saved DQN model |
| **Random** | Random feasible action each step |
| **Manual** | You choose each action via buttons or keyboard |

If no trained model exists, Q-Learning / DQN fall back to random feasible actions.

### Simulation controls

| Button | Action |
|--------|--------|
| **▶ Run** | Auto-step at the speed set by the slider |
| **⏸ Pause** | Stop auto-run (shown while running) |
| **⟳ Step** | Advance one step using the selected policy (not available in Manual mode) |
| **⟲ Reset** | Reset environment, counters, and log |

**Speed slider:** Step delay 200–2000 ms (lower = faster).

### Manual policy — recommended workflow

Manual mode lets you step through the MDP one action at a time and understand why certain actions are blocked.

**Step-by-step:**

1. Select **Manual** from the Policy dropdown (or click any action button — policy auto-switches).
2. The manual panel highlights in blue and shows which actions are currently available.
3. Click a **coloured** button or press keyboard keys **0–6**:

| Key | Action | When available |
|-----|--------|----------------|
| `0` | **Idle** | Always |
| `1` | **Active TX** | Jammer IDLE, data > 0, energy ≥ 1 |
| `2` | **Harvest** | Jammer ACTIVE |
| `3` | **Backscatter** | Jammer ACTIVE, data > 0 |
| `4` | **RA-0** | Jammer ACTIVE, data > 0 |
| `5` | **RA-1** | Jammer ACTIVE, data > 0 |
| `6` | **RA-2** | Jammer ACTIVE, data > 0 |

4. Grey buttons are **blocked** for the current state — hover for the reason.
5. Each click advances one time slot: reward is applied, queues update, jammer may transition, new packets may arrive.
6. Use **⟲ Reset** to start over.

**Button layout:**

- **Jammer IDLE row:** Idle, Active TX
- **Jammer ACTIVE row:** Harvest, Backscatter, RA-0, RA-1, RA-2

**Tips:**
- Pause auto-run before switching to Manual.
- Training locks manual controls until complete.
- Compare your choices with **View Q-Table** after Q-Learning training.

### Training from the GUI

| Button | Description |
|--------|-------------|
| **Train Q-Learning** | Trains 200k steps in background (~1 min), saves Q-table to `results/models/` |
| **Train DQN** | Trains 5k steps in background (requires TensorFlow), saves model to `results/models/` |
| **View Q-Table** | Opens scrollable table of all Q-values (requires trained Q-table) |

Progress appears in the status label (top-right).

---

## Running Tests

```bash
pytest tests/ -v
```

| Test file | Covers |
|-----------|--------|
| `test_environment.py` | State encoding, rewards, action feasibility |
| `test_q_learning.py` | Q-table updates, epsilon decay, save/load |
| `test_dqn.py` | Replay buffer, network, training step |

---

## Customising Parameters

Edit `parameters.py`:

```python
data_queue_capacity = 10      # Max data queue size
energy_queue_capacity = 10    # Max energy queue size
jammer_idle_probability = 0.1   # P(jammer → idle) each slot
packet_arrival_rate = 3         # Poisson mean arrivals per slot
q_learning_training_steps = 1_000_000
dqn_training_steps = 1_000_000
```

> Changing queue sizes changes `total_states` (242 by default). Saved Q-tables and DQN models must be retrained.

---

## Expected Results

| Metric | Q-Learning | DQN |
|--------|-----------|-----|
| Avg throughput | ~2.3–2.5 pkts/slot | ~2.2–2.4 pkts/slot |
| Convergence | ~200k–1M steps | ~400k–1M steps |
| Training time | ~1–2 min | ~5–15 min (GPU helps) |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| GUI won't start / crashes on load | Ensure Python 3.9+; run `python gui_app.py` from project root |
| DQN policy grey / random behaviour | Train DQN first or run `python train.py --agent dqn` |
| Q-Table viewer empty | Train Q-Learning first |
| `TensorFlow not found` | `pip install tensorflow` |
| Tests fail on DQN | Install TensorFlow; DQN tests need it |
| Manual buttons all grey | Select **Manual** policy; pause simulation |
| Can't find model files | Check `results/models/` (not just `results/`) |

---

## Further Reading

- [README.md](README.md) — project overview
- [documentation/results.md](documentation/results.md) — detailed results with plots
- [documentation/Deep_Q_Network.md](documentation/Deep_Q_Network.md) — DQN documentation
- [documentation/Q_Learning.md](documentation/Q_Learning.md) — Q-Learning documentation
