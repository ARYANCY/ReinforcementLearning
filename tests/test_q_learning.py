

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from q_learning_agent import QLearningAgent
from parameters import total_states, total_actions, q_learning_learning_rate, q_learning_discount_factor, initial_exploration_rate, minimum_exploration_rate


class TestQLearningAgent:

    def test_q_matrix_shape(self):
        agent = QLearningAgent(seed=0)
        assert agent.q_matrix.shape == (total_states, total_actions)

    def test_q_matrix_initialised_zero(self):
        agent = QLearningAgent(seed=0)
        assert np.all(agent.q_matrix == 0)

    def test_select_action_feasible(self):
        agent = QLearningAgent(seed=7)
        agent.environment.reset()
        for _ in range(200):
            state = agent.environment.get_state()
            possible = agent.environment.get_possible_actions()
            action = agent.select_action(state, possible)
            assert action in possible

    def test_bellman_update_changes_q(self):
        agent = QLearningAgent(seed=0)
        state = 0
        action = 0
        reward = 3
        next_state = 1
        next_possible = [0, 1]
        old_q = agent.q_matrix[state][action]
        agent.update_q_matrix(state, action, reward, next_state, next_possible)
        new_q = agent.q_matrix[state][action]
        assert new_q != old_q

    def test_bellman_update_formula(self):
        agent = QLearningAgent(seed=0)
        agent.q_matrix[1][0] = 5.0
        state, action, reward, next_state = 0, 0, 2, 1
        best_next = 5.0
        expected = 0 + q_learning_learning_rate * (reward + q_learning_discount_factor * best_next - 0)
        agent.update_q_matrix(state, action, reward, next_state, [0])
        assert abs(agent.q_matrix[state][action] - expected) < 1e-9

    def test_epsilon_decay(self):
        agent = QLearningAgent(seed=0)
        eps0 = agent.current_exploration_rate
        agent.decay_exploration_rate()
        assert agent.current_exploration_rate < eps0

    def test_epsilon_floor(self):
        agent = QLearningAgent(seed=0)
        agent.current_exploration_rate = minimum_exploration_rate * 0.5
        agent.decay_exploration_rate()
        assert agent.current_exploration_rate == minimum_exploration_rate

    def test_short_training_runs_without_error(self):
        from parameters import q_learning_training_steps as T_orig
        import parameters as P
        P.q_learning_training_steps = 500
        P.logging_interval = 100

        agent = QLearningAgent(seed=0)
        agent.train()

        P.q_learning_training_steps = T_orig
        P.logging_interval = 10000

    def test_q_table_save_load(self, tmp_path):
        agent = QLearningAgent(seed=0)
        agent.q_matrix[0][0] = 42.0
        path = str(tmp_path / "test_q.npy")
        agent.save_q_table(path)

        agent2 = QLearningAgent(seed=0)
        agent2.load_q_table(path)
        assert agent2.q_matrix[0][0] == 42.0

    def test_evaluate_returns_float(self):
        agent = QLearningAgent(seed=0)
        result = agent.evaluate_policy(num_evaluation_steps=200)
        assert isinstance(result, float)
        assert result >= 0.0
