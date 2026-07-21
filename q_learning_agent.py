
import numpy as np
import os
import csv
import time

from environment import Environment
import parameters
from parameters import (
    total_states,
    total_actions,
    q_learning_learning_rate,
    q_learning_discount_factor,
    initial_exploration_rate,
    minimum_exploration_rate,
    exploration_decay_rate,
    evaluation_window_size,
    results_directory,
)


class QLearningAgent:
    def __init__(self, seed: int = 42):
        self.environment = Environment(seed=seed)
        self.q_matrix = np.zeros((total_states, total_actions))
        self.current_exploration_rate = initial_exploration_rate
        self.reward_history = []
        self.average_reward_log = []
        os.makedirs(results_directory, exist_ok=True)

    def select_action(self, state_index: int, possible_actions: list) -> int:
        if np.random.random() < self.current_exploration_rate:
            return np.random.choice(possible_actions)
        q_values = {action: self.q_matrix[state_index][action] for action in possible_actions}
        return max(q_values, key=q_values.get)

    def update_q_matrix(self, state_index, action, reward, next_state_index, next_possible_actions):
        best_next_q_value = max(self.q_matrix[next_state_index][action] for action in next_possible_actions)
        temporal_difference_target = reward + q_learning_discount_factor * best_next_q_value
        temporal_difference_error = temporal_difference_target - self.q_matrix[state_index][action]
        self.q_matrix[state_index][action] += q_learning_learning_rate * temporal_difference_error

    def decay_exploration_rate(self):
        self.current_exploration_rate = max(minimum_exploration_rate, self.current_exploration_rate * exploration_decay_rate)

    def train(self):
        state_index = self.environment.reset()
        rolling_rewards = []
        print("=" * 60)
        print(" Q-Learning Training")
        print(f"  Steps     : {parameters.q_learning_training_steps:,}")
        print(f"  lr={q_learning_learning_rate}  gamma={q_learning_discount_factor}  epsilon-start={initial_exploration_rate}")
        print("=" * 60)
        start_time = time.time()
        for step in range(1, parameters.q_learning_training_steps + 1):
            possible_actions = self.environment.get_possible_actions()
            action = self.select_action(state_index, possible_actions)
            reward, next_state_index = self.environment.perform_action(action)
            next_possible_actions = self.environment.get_possible_actions()
            self.update_q_matrix(state_index, action, reward, next_state_index, next_possible_actions)
            self.decay_exploration_rate()
            rolling_rewards.append(reward)
            self.reward_history.append(reward)
            if len(rolling_rewards) > evaluation_window_size:
                rolling_rewards.pop(0)
            if step % parameters.logging_interval == 0:
                average_reward = np.mean(rolling_rewards)
                self.average_reward_log.append((step, average_reward))
                elapsed_time = time.time() - start_time
                print(f"  Step {step:>8,} | epsilon={self.current_exploration_rate:.4f} | Avg reward (last {evaluation_window_size}): {average_reward:.4f} | Elapsed: {elapsed_time:.1f}s")
            state_index = next_state_index
        self._save_training_results()
        print("\nTraining complete.")
        print(f"  Final avg reward: {np.mean(self.reward_history[-evaluation_window_size:]):.4f}")

    def save_q_table(self, file_path: str = None):
        file_path = file_path or os.path.join(results_directory, "q_table.npy")
        np.save(file_path, self.q_matrix)
        print(f"  Q-table saved -> {file_path}")

    def load_q_table(self, file_path: str):
        self.q_matrix = np.load(file_path)
        print(f"  Q-table loaded <- {file_path}")

    def _save_training_results(self):
        csv_path = os.path.join(results_directory, "q_learning_log.csv")
        with open(csv_path, "w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["step", "avg_reward"])
            csv_writer.writerows(self.average_reward_log)
        print(f"  Training log  -> {csv_path}")
        self.save_q_table()

    def evaluate_policy(self, num_evaluation_steps: int = 50000) -> float:
        state_index = self.environment.reset()
        evaluation_rewards = []
        for _ in range(num_evaluation_steps):
            possible_actions = self.environment.get_possible_actions()
            action = self.select_action(state_index, possible_actions)
            reward, next_state_index = self.environment.perform_action(action)
            evaluation_rewards.append(reward)
            state_index = next_state_index
        average_reward = float(np.mean(evaluation_rewards))
        print(f"  Greedy eval ({num_evaluation_steps:,} steps): avg reward = {average_reward:.4f}")
        return average_reward


if __name__ == "__main__":
    agent = QLearningAgent(seed=0)
    agent.train()
    agent.evaluate_policy()
