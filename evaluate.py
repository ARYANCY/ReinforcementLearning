
import argparse
import numpy as np
import os

from environment import Environment, decode_state
from parameters import total_states, total_actions, results_directory, energy_queue_capacity, data_queue_capacity


def parse_command_line_arguments():
    parser = argparse.ArgumentParser(description="Evaluate a trained RL agent")
    parser.add_argument("--agent", choices=["q", "dqn"], required=True)
    parser.add_argument("--model", type=str, help="Path to saved model (npy for Q, directory for DQN)")
    parser.add_argument("--steps", type=int, default=50000, help="Evaluation steps (default: 50000)")
    parser.add_argument("--heatmap", action="store_true", help="Render greedy-policy heatmap")
    return parser.parse_args()


def evaluate_agent(agent, environment, number_of_steps: int):
    from environment import encode_state
    state_index = environment.reset()
    rewards, actions_taken = [], []
    for _ in range(number_of_steps):
        possible_actions = environment.get_possible_actions()
        if hasattr(agent, "q_matrix"):
            q_values = {action: agent.q_matrix[state_index][action] for action in possible_actions}
            action = max(q_values, key=q_values.get)
        else:
            import numpy as np
            from deep_q_agent import DQNAgent
            state_vector = DQNAgent._one_hot_encode_state(state_index)
            action = agent.select_action(state_vector, possible_actions)
        reward, next_state_index = environment.perform_action(action)
        rewards.append(reward)
        actions_taken.append(action)
        state_index = next_state_index
    average_reward = np.mean(rewards)
    print(f"\nEvaluation ({number_of_steps:,} steps)")
    print(f"  Average throughput : {average_reward:.4f} packets/slot")
    print(f"  Action distribution:")
    action_labels = ["Idle", "Active-TX", "Harvest", "Backscatter", "RA-0", "RA-1", "RA-2"]
    counts = np.bincount(actions_taken, minlength=total_actions)
    for action_index, (label, count) in enumerate(zip(action_labels, counts)):
        print(f"    [{action_index}] {label:12s}: {count:7,}  ({100*count/number_of_steps:.1f}%)")
    return average_reward


def render_policy_heatmap(q_matrix):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    action_labels = ["Idle", "Active-TX", "Harvest", "Backscatter", "RA-0", "RA-1", "RA-2"]
    colormap = plt.colormaps.get_cmap("tab10").resampled(total_actions)
    normalization = mcolors.BoundaryNorm(np.arange(-0.5, total_actions), total_actions)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for jammer_state, ax in enumerate(axes):
        grid = np.zeros((data_queue_capacity + 1, energy_queue_capacity + 1), dtype=int)
        for data_queue_level in range(data_queue_capacity + 1):
            for energy_queue_level in range(energy_queue_capacity + 1):
                from environment import encode_state
                state_index = encode_state(jammer_state, data_queue_level, energy_queue_level)
                grid[data_queue_level, energy_queue_level] = int(np.argmax(q_matrix[state_index]))
        image = ax.imshow(grid, cmap=colormap, norm=normalization, origin="lower", aspect="auto")
        ax.set_xlabel("Energy queue (e)", fontsize=11)
        ax.set_ylabel("Data queue (d)", fontsize=11)
        ax.set_title(f"Greedy Policy — Jammer {'ACTIVE' if jammer_state else 'IDLE'}", fontsize=12)
    scalar_mappable = plt.cm.ScalarMappable(cmap=colormap, norm=normalization)
    scalar_mappable.set_array([])
    colorbar = figure.colorbar(scalar_mappable, ax=axes, ticks=range(total_actions), shrink=0.8)
    colorbar.ax.set_yticklabels(action_labels)
    output_file_path = os.path.join(results_directory, "policy_heatmap.png")
    os.makedirs(results_directory, exist_ok=True)
    figure.savefig(output_file_path, dpi=150, bbox_inches="tight")
    print(f"  Heatmap saved -> {output_file_path}")
    plt.show()


if __name__ == "__main__":
    arguments = parse_command_line_arguments()
    environment = Environment(seed=99)
    if arguments.agent == "q":
        from q_learning_agent import QLearningAgent
        agent = QLearningAgent()
        model_path = arguments.model or os.path.join(results_directory, "q_table.npy")
        agent.load_q_table(model_path)
        evaluate_agent(agent, environment, arguments.steps)
        if arguments.heatmap:
            render_policy_heatmap(agent.q_matrix)
    elif arguments.agent == "dqn":
        from deep_q_agent import DQNAgent
        agent = DQNAgent()
        model_path = arguments.model or os.path.join(results_directory, "dqn_model.keras")
        agent.load_model(model_path)
        evaluate_agent(agent, environment, arguments.steps)
