import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from environment import Environment, encode_state
import parameters
from parameters import (
    total_actions,
    data_queue_capacity,
    energy_queue_capacity,
    results_directory,
    plots_directory,
    logs_directory,
    models_directory,
)


def get_existing_file_path(filename: str, preferred_dir: str, fallback_dir: str):
    primary = os.path.join(preferred_dir, filename)
    if os.path.exists(primary):
        return primary
    fallback = os.path.join(fallback_dir, filename)
    if os.path.exists(fallback):
        return fallback
    return primary


def load_log_data(csv_filename: str):
    file_path = get_existing_file_path(csv_filename, logs_directory, results_directory)
    if not os.path.exists(file_path):
        return None, None, None, None
    steps, rewards, avg_rewards, cum_rewards = [], [], [], []
    with open(file_path, "r") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            steps.append(int(row["step"]))
            rewards.append(float(row.get("reward", row.get("avg_reward", 0))))
            avg_rewards.append(float(row["avg_reward"]))
            cum_rewards.append(float(row.get("cumulative_reward", 0)))
    return np.array(steps), np.array(rewards), np.array(avg_rewards), np.array(cum_rewards)


def generate_01_reward_vs_training_steps():
    """01_reward_vs_training_steps.png"""
    os.makedirs(plots_directory, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)

    q_steps, q_rewards, q_avg, _ = load_log_data("q_learning_log.csv")
    dqn_steps, dqn_rewards, dqn_avg, _ = load_log_data("dqn_log.csv")

    if q_steps is not None and len(q_steps) > 0:
        ax.plot(q_steps, q_rewards, alpha=0.35, color="#1f77b4", linestyle="--", linewidth=1, label="Q-Learning (Sampled)")
        ax.plot(q_steps, q_avg, color="#1f77b4", linewidth=2.5, label="Q-Learning (Smoothed)")

    if dqn_steps is not None and len(dqn_steps) > 0:
        ax.plot(dqn_steps, dqn_rewards, alpha=0.35, color="#ff7f0e", linestyle="--", linewidth=1, label="DQN (Sampled)")
        ax.plot(dqn_steps, dqn_avg, color="#ff7f0e", linewidth=2.5, label="DQN (Smoothed)")

    ax.set_xlabel("Training Steps", fontsize=12, fontweight="bold")
    ax.set_ylabel("Immediate Reward (packets/slot)", fontsize=12, fontweight="bold")
    ax.set_title("01. Immediate Reward vs. Training Steps", fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=10, loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    out_path = os.path.join(plots_directory, "01_reward_vs_training_steps.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Generated -> {out_path}")


def generate_02_average_throughput_vs_training_steps():
    """02_average_throughput_vs_training_steps.png"""
    os.makedirs(plots_directory, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)

    q_steps, _, q_avg, _ = load_log_data("q_learning_log.csv")
    dqn_steps, _, dqn_avg, _ = load_log_data("dqn_log.csv")

    if q_steps is not None and len(q_steps) > 0:
        ax.plot(q_steps, q_avg, color="#007acc", linewidth=2.5, label="Tabular Q-Learning")
    if dqn_steps is not None and len(dqn_steps) > 0:
        ax.plot(dqn_steps, dqn_avg, color="#7c3aed", linewidth=2.5, label="Deep Q-Network (DQN)")

    ax.set_xlabel("Training Steps", fontsize=12, fontweight="bold")
    ax.set_ylabel("Average Throughput (packets/slot)", fontsize=12, fontweight="bold")
    ax.set_title("02. Average Throughput vs. Training Steps", fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=11, loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    out_path = os.path.join(plots_directory, "02_average_throughput_vs_training_steps.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Generated -> {out_path}")


def generate_03_cumulative_reward_vs_training_steps():
    """03_cumulative_reward_vs_training_steps.png"""
    os.makedirs(plots_directory, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)

    q_steps, _, _, q_cum = load_log_data("q_learning_log.csv")
    dqn_steps, _, _, dqn_cum = load_log_data("dqn_log.csv")

    if q_steps is not None and len(q_steps) > 0:
        ax.plot(q_steps, q_cum, color="#2ecc71", linewidth=2.5, label="Q-Learning Cumulative Delivered")
    if dqn_steps is not None and len(dqn_steps) > 0:
        ax.plot(dqn_steps, dqn_cum, color="#e67e22", linewidth=2.5, label="DQN Cumulative Delivered")

    ax.set_xlabel("Training Steps", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cumulative Delivered Packets", fontsize=12, fontweight="bold")
    ax.set_title("03. Cumulative Reward vs. Training Steps", fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=11, loc="upper left", frameon=True, facecolor="white", edgecolor="#cccccc")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    out_path = os.path.join(plots_directory, "03_cumulative_reward_vs_training_steps.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Generated -> {out_path}")


def generate_04_q_table_heatmap(q_matrix=None):
    """04_q_table_heatmap.png"""
    os.makedirs(plots_directory, exist_ok=True)
    q_table_path = get_existing_file_path("q_table.npy", models_directory, results_directory)
    if q_matrix is None:
        if os.path.exists(q_table_path):
            q_matrix = np.load(q_table_path)
        else:
            q_matrix = np.zeros((parameters.total_states, total_actions))

    action_labels = ["Idle", "Active-TX", "Harvest", "Backscatter", "RA-0", "RA-1", "RA-2"]
    colors = ["#7f8c8d", "#1a73e8", "#1e8449", "#d4ac0d", "#e67e22", "#e74c3c", "#8e44ad"]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, total_actions), total_actions)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True, dpi=150)
    for jammer_state, ax in enumerate(axes):
        grid = np.zeros((data_queue_capacity + 1, energy_queue_capacity + 1), dtype=int)
        for d in range(data_queue_capacity + 1):
            for e in range(energy_queue_capacity + 1):
                state_idx = encode_state(jammer_state, d, e)
                grid[d, e] = int(np.argmax(q_matrix[state_idx]))

        im = ax.imshow(grid, cmap=cmap, norm=norm, origin="lower", aspect="auto")
        ax.set_xlabel("Energy Queue (e)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Data Queue (d)", fontsize=11, fontweight="bold")
        ax.set_title(f"Jammer {'ACTIVE' if jammer_state else 'IDLE'} (j={jammer_state})", fontsize=12, fontweight="bold")
        ax.set_xticks(range(0, energy_queue_capacity + 1, 2))
        ax.set_yticks(range(0, data_queue_capacity + 1, 2))
        ax.grid(False)

    fig.suptitle("04. Optimal Greedy Policy Heatmap (Q-Table)", fontsize=14, fontweight="bold", y=1.02)
    cbar = fig.colorbar(im, ax=axes, ticks=range(total_actions), shrink=0.85, pad=0.03)
    cbar.ax.set_yticklabels(action_labels, fontsize=10, fontweight="bold")

    out_path = os.path.join(plots_directory, "04_q_table_heatmap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Generated -> {out_path}")


def generate_05_action_selection_distribution(actions_taken=None):
    """05_action_selection_distribution.png"""
    os.makedirs(plots_directory, exist_ok=True)
    action_labels = ["Idle", "Active-TX", "Harvest", "Backscatter", "RA-0", "RA-1", "RA-2"]
    if actions_taken is None:
        env = Environment(seed=42)
        q_table_path = get_existing_file_path("q_table.npy", models_directory, results_directory)
        if os.path.exists(q_table_path):
            q_matrix = np.load(q_table_path)
            state_idx = env.reset()
            actions_taken = []
            for _ in range(50000):
                possible = env.get_possible_actions()
                q_vals = {a: q_matrix[state_idx][a] for a in possible}
                action = max(q_vals, key=q_vals.get)
                reward, next_idx = env.perform_action(action)
                actions_taken.append(action)
                state_idx = next_idx
        else:
            actions_taken = np.random.choice(total_actions, size=50000)

    counts = np.bincount(actions_taken, minlength=total_actions)
    percentages = 100.0 * counts / np.sum(counts)

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    bar_colors = ["#7f8c8d", "#1a73e8", "#1e8449", "#d4ac0d", "#e67e22", "#e74c3c", "#8e44ad"]

    bars = ax.bar(action_labels, percentages, color=bar_colors, edgecolor="#333333", linewidth=1, width=0.6)

    for bar, pct, count in zip(bars, percentages, counts):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.0, f"{pct:.1f}%\n({count:,})", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Selection Percentage (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Action Type", fontsize=12, fontweight="bold")
    ax.set_title("05. Action Selection Distribution During Evaluation", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0, max(percentages) + 12)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()

    out_path = os.path.join(plots_directory, "05_action_selection_distribution.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Generated -> {out_path}")


def generate_all_plots(q_matrix=None, actions_taken=None):
    print("=" * 60)
    print(" Generating All 5 Enhanced Plots...")
    print("=" * 60)
    generate_01_reward_vs_training_steps()
    generate_02_average_throughput_vs_training_steps()
    generate_03_cumulative_reward_vs_training_steps()
    generate_04_q_table_heatmap(q_matrix)
    generate_05_action_selection_distribution(actions_taken)
    print("=" * 60)
    print(f" All 5 plots generated successfully in {plots_directory}/")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_plots()
