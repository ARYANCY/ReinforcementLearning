
import numpy as np
import os
import csv
import time
import random
from collections import deque

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("[WARNING] TensorFlow not found. DQN agent requires TensorFlow >= 2.x.")

from environment import Environment
import parameters
from parameters import (
    total_states,
    total_actions,
    dqn_learning_rate,
    dqn_discount_factor,
    initial_exploration_rate,
    minimum_exploration_rate,
    exploration_decay_rate,
    replay_buffer_capacity,
    training_batch_size,
    target_network_update_frequency,
    dqn_hidden_layer_sizes,
    evaluation_window_size,
    results_directory,
)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add_experience(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample_batch(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


def build_q_network(input_dimension: int, output_dimension: int, hidden_layer_sizes: list, learning_rate: float) -> "keras.Model":
    inputs = keras.Input(shape=(input_dimension,), name="state")
    x = inputs
    for i, units in enumerate(hidden_layer_sizes):
        x = layers.Dense(units, activation="relu", name=f"fc{i+1}")(x)
    outputs = layers.Dense(output_dimension, activation="linear", name="q_values")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model


class DQNAgent:
    def __init__(self, seed: int = 42):
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow is required to run DQNAgent.")
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)
        self.environment = Environment(seed=seed)
        self.current_exploration_rate = initial_exploration_rate
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)
        self.online_network = build_q_network(total_states, total_actions, dqn_hidden_layer_sizes, dqn_learning_rate)
        self.target_network = build_q_network(total_states, total_actions, dqn_hidden_layer_sizes, dqn_learning_rate)
        self._synchronize_target_network()
        self.reward_history = []
        self.average_reward_log = []
        self.loss_log = []
        os.makedirs(results_directory, exist_ok=True)

    @staticmethod
    def _one_hot_encode_state(state_index: int) -> np.ndarray:
        one_hot_vector = np.zeros(total_states, dtype=np.float32)
        one_hot_vector[state_index] = 1.0
        return one_hot_vector

    def select_action(self, state_vector: np.ndarray, possible_actions: list) -> int:
        if np.random.random() < self.current_exploration_rate:
            return np.random.choice(possible_actions)
        q_values = self.online_network(state_vector[np.newaxis], training=False)[0].numpy()
        mask = np.full(total_actions, -np.inf)
        for action in possible_actions:
            mask[action] = q_values[action]
        return int(np.argmax(mask))

    def train_step(self) -> float:
        if len(self.replay_buffer) < training_batch_size:
            return 0.0
        states, actions, rewards, next_states, dones = self.replay_buffer.sample_batch(training_batch_size)
        next_q_values = self.target_network(next_states, training=False).numpy()
        targets = rewards + (1 - dones) * dqn_discount_factor * np.max(next_q_values, axis=1)
        with tf.GradientTape() as tape:
            q_predictions = self.online_network(states, training=True)
            action_mask = tf.one_hot(actions, total_actions)
            q_taken = tf.reduce_sum(q_predictions * action_mask, axis=1)
            loss = tf.reduce_mean(tf.square(targets - q_taken))
        gradients = tape.gradient(loss, self.online_network.trainable_variables)
        self.online_network.optimizer.apply_gradients(zip(gradients, self.online_network.trainable_variables))
        return float(loss)

    def _synchronize_target_network(self):
        self.target_network.set_weights(self.online_network.get_weights())

    def train(self):
        state_index = self.environment.reset()
        state_vector = self._one_hot_encode_state(state_index)
        rolling_rewards = []
        print("=" * 60)
        print(" Deep Q-Network (DQN) Training")
        print(f"  Steps         : {parameters.dqn_training_steps:,}")
        print(f"  Replay buffer : {replay_buffer_capacity:,}  Batch: {training_batch_size}")
        print(f"  Target sync   : every {target_network_update_frequency} steps")
        print(f"  Network       : {total_states} -> {dqn_hidden_layer_sizes} -> {total_actions}")
        print("=" * 60)
        start_time = time.time()
        for step in range(1, parameters.dqn_training_steps + 1):
            possible_actions = self.environment.get_possible_actions()
            action = self.select_action(state_vector, possible_actions)
            reward, next_state_index = self.environment.perform_action(action)
            next_state_vector = self._one_hot_encode_state(next_state_index)
            done = False
            self.replay_buffer.add_experience(state_vector, action, reward, next_state_vector, done)
            loss = self.train_step()
            self.current_exploration_rate = max(minimum_exploration_rate, self.current_exploration_rate * exploration_decay_rate)
            if step % target_network_update_frequency == 0:
                self._synchronize_target_network()
            rolling_rewards.append(reward)
            self.reward_history.append(reward)
            if len(rolling_rewards) > evaluation_window_size:
                rolling_rewards.pop(0)
            if loss > 0:
                self.loss_log.append(loss)
            if step % parameters.logging_interval == 0:
                average_reward = np.mean(rolling_rewards)
                self.average_reward_log.append((step, average_reward))
                elapsed_time = time.time() - start_time
                average_loss = np.mean(self.loss_log[-1000:]) if self.loss_log else 0
                print(f"  Step {step:>8,} | epsilon={self.current_exploration_rate:.4f} | Avg reward: {average_reward:.4f} | Loss: {average_loss:.5f} | Elapsed: {elapsed_time:.1f}s")
            state_index = next_state_index
            state_vector = next_state_vector
        self._save_training_results()
        print("\nDQN training complete.")
        print(f"  Final avg reward: {np.mean(self.reward_history[-evaluation_window_size:]):.4f}")

    def save_model(self, file_path: str = None):
        file_path = file_path or os.path.join(results_directory, "dqn_model.keras")
        self.online_network.save(file_path)
        print(f"  DQN model saved -> {file_path}")

    def load_model(self, file_path: str):
        self.online_network = keras.models.load_model(file_path)
        self._synchronize_target_network()
        print(f"  DQN model loaded <- {file_path}")

    def _save_training_results(self):
        csv_path = os.path.join(results_directory, "dqn_log.csv")
        with open(csv_path, "w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["step", "avg_reward"])
            csv_writer.writerows(self.average_reward_log)
        print(f"  Training log  -> {csv_path}")
        self.save_model()

    def evaluate_policy(self, num_evaluation_steps: int = 50000) -> float:
        state_index = self.environment.reset()
        state_vector = self._one_hot_encode_state(state_index)
        evaluation_rewards = []
        for _ in range(num_evaluation_steps):
            possible_actions = self.environment.get_possible_actions()
            action = self.select_action(state_vector, possible_actions)
            reward, next_state_index = self.environment.perform_action(action)
            evaluation_rewards.append(reward)
            state_vector = self._one_hot_encode_state(next_state_index)
        average_reward = float(np.mean(evaluation_rewards))
        print(f"  DQN greedy eval ({num_evaluation_steps:,} steps): avg reward = {average_reward:.4f}")
        return average_reward


if __name__ == "__main__":
    agent = DQNAgent(seed=0)
    agent.train()
    agent.evaluate_policy()
