import argparse
import numpy as np
import os

from environment import Environment, decode_state
from parameters import total_states, total_actions, results_directory, models_directory, energy_queue_capacity, data_queue_capacity
from generate_plots import (
    generate_all_plots,
    generate_04_q_table_heatmap,
    generate_05_action_selection_distribution,
)


def parse_command_line_arguments():
    parser = argparse.ArgumentParser(description="Evaluate a trained RL agent")
    parser.add_argument("--agent", choices=["q", "dqn"], default="q", help="Agent to evaluate (default: q)")
    parser.add_argument("--model", type=str, help="Path to saved model")
    parser.add_argument("--steps", type=int, default=50000, help="Evaluation steps (default: 50000)")
    parser.add_argument("--plot", action="store_true", help="Generate all 5 evaluation plots")
    parser.add_argument("--heatmap", action="store_true", help="Render Q-table heatmap")
    return parser.parse_args()


def evaluate_agent(agent, environment, number_of_steps: int):
    state_index = environment.reset()
    rewards, actions_taken = [], []
    for _ in range(number_of_steps):
        possible_actions = environment.get_possible_actions()
        if hasattr(agent, "q_matrix"):
            q_values = {action: agent.q_matrix[state_index][action] for action in possible_actions}
            action = max(q_values, key=q_values.get)
        else:
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
    return average_reward, actions_taken


if __name__ == "__main__":
    arguments = parse_command_line_arguments()
    environment = Environment(seed=99)
    actions_taken = None
    q_matrix = None

    if arguments.agent == "q":
        from q_learning_agent import QLearningAgent
        agent = QLearningAgent()
        model_path = arguments.model or os.path.join(models_directory, "q_table.npy")
        if not os.path.exists(model_path):
            fallback_path = os.path.join(results_directory, "q_table.npy")
            if os.path.exists(fallback_path):
                model_path = fallback_path

        if os.path.exists(model_path):
            agent.load_q_table(model_path)
            q_matrix = agent.q_matrix
            _, actions_taken = evaluate_agent(agent, environment, arguments.steps)
            if arguments.heatmap:
                generate_04_q_table_heatmap(agent.q_matrix)
        else:
            print(f"[WARNING] Model file not found at {model_path}. Train the agent first.")
    elif arguments.agent == "dqn":
        from deep_q_agent import DQNAgent, TENSORFLOW_AVAILABLE
        if TENSORFLOW_AVAILABLE:
            agent = DQNAgent()
            model_path = arguments.model or os.path.join(models_directory, "dqn_model.keras")
            if not os.path.exists(model_path):
                fallback_path = os.path.join(results_directory, "dqn_model.keras")
                if os.path.exists(fallback_path):
                    model_path = fallback_path

            if os.path.exists(model_path):
                agent.load_model(model_path)
                _, actions_taken = evaluate_agent(agent, environment, arguments.steps)
            else:
                print(f"[WARNING] Model file not found at {model_path}. Train the agent first.")

    if arguments.plot:
        generate_all_plots(q_matrix=q_matrix, actions_taken=actions_taken)
