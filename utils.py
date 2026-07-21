
import numpy as np
import os
import json
from typing import List, Tuple, Dict


def set_global_seed(seed: int):
    import random
    np.random.seed(seed)
    random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def compute_moving_average(values: List[float], window_size: int) -> np.ndarray:
    result = np.zeros(len(values))
    for i in range(len(values)):
        lower_index = max(0, i - window_size + 1)
        result[i] = np.mean(values[lower_index : i + 1])
    return result


def get_action_name(action_index: int) -> str:
    action_names = {0: "Idle", 1: "Active-TX", 2: "Harvest", 3: "Backscatter", 4: "RA-0", 5: "RA-1", 6: "RA-2"}
    return action_names.get(action_index, f"Action-{action_index}")


def print_q_table_row(q_matrix: np.ndarray, state_index: int):
    from environment import decode_state
    jammer_state, data_queue_level, energy_queue_level = decode_state(state_index)
    print(f"State ({jammer_state},{data_queue_level},{energy_queue_level}) -> idx {state_index}:")
    for action_index, q_value in enumerate(q_matrix[state_index]):
        print(f"  {get_action_name(action_index):12s}: {q_value:+.4f}")


def compute_performance_metrics(reward_history: List[float], window_size: int = 1000) -> Dict[str, float]:
    reward_array = np.array(reward_history)
    return {
        "total_steps": len(reward_array),
        "mean_reward": float(np.mean(reward_array)),
        "std_reward": float(np.std(reward_array)),
        "max_reward": float(np.max(reward_array)),
        "final_avg": float(np.mean(reward_array[-window_size:])),
        "zero_reward_pct": float(100.0 * np.mean(reward_array == 0)),
    }


def save_metrics_to_json(metrics: Dict, file_path: str):
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    with open(file_path, "w") as json_file:
        json.dump(metrics, json_file, indent=2)
    print(f"  Metrics saved -> {file_path}")


def plot_reward_curve(steps: List[int], average_rewards: List[float], plot_title: str = "Reward Curve", save_file_path: str = None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot_reward_curve] matplotlib not available; skipping plot.")
        return
    figure, axes = plt.subplots(figsize=(9, 4))
    axes.plot(steps, average_rewards, linewidth=2, color="steelblue")
    axes.set_xlabel("Training Step")
    axes.set_ylabel("Avg Reward (packets/slot)")
    axes.set_title(plot_title)
    axes.grid(True, alpha=0.3)
    figure.tight_layout()
    if save_file_path:
        figure.savefig(save_file_path, dpi=150)
        print(f"  Plot saved -> {save_file_path}")
    plt.show()
    return figure


def plot_agent_comparison(logs: List[Tuple[str, List[int], List[float]]], save_file_path: str = None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot_agent_comparison] matplotlib not available.")
        return
    colors = ["steelblue", "darkorange", "seagreen", "firebrick"]
    figure, axes = plt.subplots(figsize=(10, 5))
    for i, (label, steps, rewards) in enumerate(logs):
        axes.plot(steps, rewards, label=label, color=colors[i % len(colors)], linewidth=2)
    axes.set_xlabel("Training Step", fontsize=12)
    axes.set_ylabel("Average Throughput (packets/slot)", fontsize=12)
    axes.set_title("RL Agent Convergence Comparison", fontsize=13)
    axes.legend(fontsize=11)
    axes.grid(True, alpha=0.3)
    figure.tight_layout()
    if save_file_path:
        figure.savefig(save_file_path, dpi=150)
        print(f"  Comparison plot saved -> {save_file_path}")
    plt.show()
    return figure
