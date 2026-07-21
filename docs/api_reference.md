# API Reference

> Module-level documentation for all Python source files.

---

## `parameters.py`

This module contains all global configuration parameters (hyperparameters, queue sizes, jammer parameters, etc.). Import these constants instead of hardcoding values elsewhere.

| Parameter | Type | Default Value | Description |
|-----------|------|---------------|-------------|
| `nu` | `float` | `0.1` | Jammer idle probability |
| `arrival_rate` | `int` | `3` | Poisson arrival rate for data packets |
| `nu_p` | `list[float]` | `[0.6, 0.2, 0.2]` | Probability distribution over jammer power levels |
| `d_t` | `int` | `4` | Packets transmitted per active transmit slot |
| `e_t` | `int` | `1` | Energy units consumed per packet in active transmit |
| `d_bj_arr` | `list[int]` | `[1, 2, 3]` | Packets delivered via backscatter per jammer power level |
| `e_hj_arr` | `list[int]` | `[1, 2, 3]` | Energy harvested per jammer power level |
| `d_ta_arr` | `list[int]` | `[2, 1, 0]` | Packets delivered via rate adaptation per level |
| `b_dagger` | `int` | `3` | Fixed backscatter rate |
| `d_queue_size` | `int` | `10` | Maximum data queue capacity |
| `e_queue_size` | `int` | `10` | Maximum energy queue capacity |
| `num_actions` | `int` | `7` | Total number of possible actions |
| `num_states` | `int` | `242` | Total number of possible states |
| `learning_rate_Q` | `float` | `0.1` | Q-Learning learning rate |
| `gamma_Q` | `float` | `0.9` | Q-Learning discount factor |
| `epsilon_start` | `float` | `1.0` | Initial exploration rate |
| `epsilon_end` | `float` | `0.01` | Minimum exploration rate |
| `epsilon_decay` | `float` | `0.9999` | Exponential decay factor per step |
| `T` | `int` | `1000000` | Q-Learning training steps |
| `learning_rate_DQN` | `float` | `0.001` | DQN Adam optimizer learning rate |
| `gamma_DQN` | `float` | `0.9` | DQN discount factor |
| `memory_size` | `int` | `2000` | Experience replay buffer capacity |
| `batch_size` | `int` | `32` | Mini-batch size for DQN training |
| `target_update_freq` | `int` | `100` | Steps between target network updates |
| `hidden_units` | `list[int]` | `[64, 64]` | Number of units per DQN hidden layer |
| `T_DQN` | `int` | `1000000` | DQN training steps |
| `log_interval` | `int` | `10000` | Steps between log outputs |
| `eval_window` | `int` | `1000` | Window size for rolling average reward |
| `results_dir` | `str` | `'results'` | Directory for saving models and logs |

---

## `environment.py`

This module defines the Markov Decision Process (MDP) environment.

### Functions
#### `encode_state(j: int, d: int, e: int) -> int`
Maps a state tuple `(j, d, e)` to a unique integer index.
- **Parameters**:
  - `j`: Jammer state (0 or 1)
  - `d`: Data queue occupancy (0 to `d_queue_size`)
  - `e`: Energy queue occupancy (0 to `e_queue_size`)
- **Returns**: Integer state index (0 to 241)

#### `decode_state(idx: int) -> tuple[int, int, int]`
Recovers the state tuple `(j, d, e)` from an integer index. Inverse of `encode_state`.
- **Parameters**: `idx` — Integer state index
- **Returns**: Tuple `(j, d, e)`

### Classes
#### `class Environment`
The main environment class.
##### `__init__(seed: int | None = None)`
Initializes the environment.
- **Parameters**: `seed` — Random seed for reproducibility (if `None`, seed is random)
##### `reset() -> int`
Resets the environment to initial state (`j=0`, `d=0`, `e=0`).
- **Returns**: Initial state index
##### `get_state() -> int`
Returns the current state index.
- **Returns**: Current state index
##### `get_possible_actions() -> list[int]`
Returns a list of feasible action indices in the current state (always includes action 0: Idle).
- **Returns**: List of feasible action indices
##### `perform_action(action: int) -> tuple[int, int]`
Executes an action and transitions the environment.
- **Parameters**: `action` — Action index
- **Returns**: Tuple `(reward, next_state_idx)`
  - `reward`: Number of packets delivered in this step
  - `next_state_idx`: Next state index
##### `state_tuple() -> tuple[int, int, int]`
Returns the current state as a tuple `(j, d, e)`.
- **Returns**: Current state tuple

---

## `q_learning_agent.py`

This module implements the tabular Q-Learning agent.

### Classes
#### `class QLearningAgent`
##### `__init__(seed: int = 42)`
Initializes the Q-table to all zeros, creates an environment, and sets up logging.
- **Parameters**: `seed` — Random seed for reproducibility
##### `select_action(state: int, possible_actions: list[int]) -> int`
Selects an action using ε-greedy strategy over feasible actions.
- **Parameters**:
  - `state`: Current state index
  - `possible_actions`: List of feasible action indices
- **Returns**: Selected action index
##### `update(state: int, action: int, reward: float, next_state: int, next_possible: list[int])`
Performs a Bellman update on the Q-table.
- **Parameters**:
  - `state`: Current state index
  - `action`: Selected action index
  - `reward`: Received reward
  - `next_state`: Next state index
  - `next_possible`: Feasible actions in next state
##### `decay_epsilon()`
Decays the exploration rate `self.epsilon` according to the decay schedule.
##### `train()`
Runs the full Q-Learning training loop for `T` steps. Logs progress every `log_interval` steps, saves Q-table and training log on completion.
- **Returns**: Tuple `(q_table, reward_history)`
##### `evaluate(n_steps: int = 50000) -> float`
Evaluates the current greedy policy for `n_steps`.
- **Parameters**: `n_steps` — Number of evaluation steps
- **Returns**: Average reward per step
##### `save_q_table(path: str | None = None)`
Saves the Q-table to a NumPy `.npy` file.
- **Parameters**: `path` — Path to save file (if `None`, uses `results_dir/q_table.npy`)
##### `load_q_table(path: str)`
Loads a Q-table from a NumPy `.npy` file.
- **Parameters**: `path` — Path to saved Q-table file

---

## `deep_q_agent.py`

This module implements the Deep Q-Network (DQN) agent.

### Classes
#### `class ReplayBuffer`
Experience replay buffer for storing and sampling past experiences.
##### `__init__(capacity: int)`
Initializes buffer with given capacity.
- **Parameters**: `capacity` — Maximum number of stored experiences
##### `push(state, action, reward, next_state, done)`
Adds a new experience to the buffer. If buffer is full, oldest experience is removed.
##### `sample(batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]`
Samples a mini-batch of experiences uniformly at random.
- **Parameters**: `batch_size` — Number of experiences to sample
- **Returns**: Tuple of NumPy arrays `(states, actions, rewards, next_states, dones)`
##### `__len__() -> int`
Returns current number of stored experiences.
- **Returns**: Number of stored experiences

### Functions
#### `build_q_network(input_dim: int, output_dim: int, hidden: list[int], lr: float)`
Builds and compiles a Keras feedforward Q-network with given dimensions.
- **Parameters**:
  - `input_dim`: Input dimension (should be `num_states`)
  - `output_dim`: Output dimension (should be `num_actions`)
  - `hidden`: List of hidden layer sizes
  - `lr`: Learning rate for Adam optimizer
- **Returns**: Compiled Keras Model

### Classes (continued)
#### `class DQNAgent`
##### `__init__(seed: int = 42)`
Initializes online and target networks, replay buffer, environment, and logging structures.
- **Parameters**: `seed` — Random seed for reproducibility
##### `select_action(state_vec: np.ndarray, possible_actions: list[int]) -> int`
Selects an action using ε-greedy strategy with infeasible action masking.
- **Parameters**:
  - `state_vec`: One-hot encoded state vector
  - `possible_actions`: List of feasible action indices
- **Returns**: Selected action index
##### `learn() -> float`
Samples a mini-batch from replay buffer and performs one gradient update step.
- **Returns**: Training loss
##### `train()`
Runs full DQN training loop for `T_DQN` steps. Logs progress every `log_interval` steps, saves model and training log on completion.
- **Returns**: Tuple `(model, reward_history)`
##### `evaluate(n_steps: int = 50000) -> float`
Evaluates current greedy policy for `n_steps`.
- **Parameters**: `n_steps` — Number of evaluation steps
- **Returns**: Average reward per step
##### `save_model(path: str | None = None)`
Saves the Keras model to file (`.keras` format).
- **Parameters**: `path` — Path to save file (if `None`, uses `results_dir/dqn_model.keras`)
##### `load_model(path: str)`
Loads a saved Keras model and synchronizes target network.
- **Parameters**: `path` — Path to saved model file
##### `_one_hot(state_idx: int) -> np.ndarray`
Converts integer state index to one-hot encoded float32 vector.
- **Parameters**: `state_idx` — Integer state index
- **Returns**: One-hot encoded vector of length `num_states`

---

## `train.py`

This module provides a command-line interface (CLI) for training agents.

### Usage
```bash
# Train Q-Learning only
python train.py --agent q

# Train DQN only
python train.py --agent dqn

# Train both Q-Learning and DQN
python train.py --agent both

# Train and evaluate after training
python train.py --agent q --eval

# Train, evaluate, and generate convergence plot
python train.py --agent both --eval --plot

# Set random seed for reproducibility
python train.py --agent q --seed 1234
```

### CLI Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--agent` | `str` | `'q'` | Agent type: `'q'` (Q-Learning), `'dqn'` (DQN), or `'both'` |
| `--eval` | `flag` | `False` | Run evaluation after training |
| `--seed` | `int` | `None` | Random seed for reproducibility |
| `--plot` | `flag` | `False` | Generate convergence plot after training |

---

## `evaluate.py`

This module provides a CLI for evaluating trained agents and generating visualizations.

### Usage
```bash
# Evaluate pre-trained Q-table
python evaluate.py --agent q --model results/q_table.npy --steps 50000

# Evaluate pre-trained DQN model
python evaluate.py --agent dqn --model results/dqn_model.keras --steps 50000

# Generate policy heatmap for Q-Learning
python evaluate.py --agent q --model results/q_table.npy --heatmap
```

### CLI Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--agent` | `str` | `'q'` | Agent type: `'q'` or `'dqn'` |
| `--model` | `str` | `None` | Path to saved model/Q-table file |
| `--steps` | `int` | `50000` | Number of evaluation steps |
| `--heatmap` | `flag` | `False` | Generate policy heatmap (Q-Learning only) |

---

## `utils.py`

This module contains shared utility functions.

### Functions
#### `set_global_seed(seed: int)`
Sets random seeds for NumPy, Python random module, and TensorFlow (if available) for reproducibility.
- **Parameters**: `seed` — Random seed value

#### `moving_average(values: list[float], window: int) -> np.ndarray`
Computes causal moving average over a list of values with given window size.
- **Parameters**:
  - `values`: List of input values
  - `window`: Window size for moving average
- **Returns**: NumPy array of moving averages (same length as input)

#### `action_name(action_idx: int) -> str`
Returns human-readable name for an action index.
- **Parameters**: `action_idx` — Action index (0 to 6)
- **Returns**: Human-readable action name string

#### `print_q_row(q_matrix: np.ndarray, state_idx: int)`
Pretty-prints a single row of the Q-table (all Q-values for one state) with decoded state label.
- **Parameters**:
  - `q_matrix`: Q-table NumPy array
  - `state_idx`: State index to print

#### `compute_metrics(reward_history: list[float], window: int = 1000) -> dict[str, float]`
Computes performance metrics from a reward history.
- **Parameters**:
  - `reward_history`: List of rewards per step
  - `window`: Window size for final average
- **Returns**: Dictionary with keys:
  - `'total_steps'`: Total number of steps
  - `'mean_reward'`: Mean reward over all steps
  - `'std_reward'`: Standard deviation of rewards
  - `'max_reward'`: Maximum single-step reward
  - `'final_avg'`: Mean reward over last `window` steps
  - `'zero_reward_pct'`: Percentage of steps with zero reward

#### `save_metrics(metrics: dict, path: str)`
Saves a metrics dictionary to JSON file.
- **Parameters**:
  - `metrics`: Dictionary to save
  - `path`: Path to save file

#### `plot_reward_curve(steps: list[int], avg_rewards: list[float], title: str, save_path: str | None = None)`
Plots a single reward convergence curve.
- **Parameters**:
  - `steps`: List of step numbers
  - `avg_rewards`: List of average rewards
  - `title`: Plot title
  - `save_path`: If provided, saves plot to file; otherwise displays plot

#### `plot_comparison(logs: list[tuple[str, list[int], list[float]]], save_path: str | None = None)`
Plots multiple reward curves on the same axes for comparison.
- **Parameters**:
  - `logs`: List of tuples `(label, steps, avg_rewards)`
  - `save_path`: If provided, saves plot to file; otherwise displays plot
