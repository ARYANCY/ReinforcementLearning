

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

tf_available = True
try:
    import tensorflow
except ImportError:
    tf_available = False

skip_no_tf = pytest.mark.skipif(not tf_available,
                                reason="TensorFlow not installed")

from parameters import total_states, total_actions, training_batch_size, replay_buffer_capacity, dqn_training_steps, logging_interval, target_network_update_frequency


class TestReplayBuffer:
    def test_push_and_len(self):
        from deep_q_agent import ReplayBuffer
        buf = ReplayBuffer(capacity=10)
        for i in range(5):
            buf.add_experience(np.zeros(total_states), i % total_actions, float(i),
                     np.ones(total_states), False)
        assert len(buf) == 5

    def test_fifo_overflow(self):
        from deep_q_agent import ReplayBuffer
        buf = ReplayBuffer(capacity=3)
        for i in range(5):
            buf.add_experience(np.zeros(total_states), 0, 0.0, np.zeros(total_states), False)
        assert len(buf) == 3

    def test_sample_shape(self):
        from deep_q_agent import ReplayBuffer
        buf = ReplayBuffer(capacity=100)
        for _ in range(50):
            buf.add_experience(np.zeros(total_states), 0, 1.0, np.ones(total_states), False)
        states, actions, rewards, next_states, dones = buf.sample_batch(16)
        assert states.shape == (16, total_states)
        assert actions.shape == (16,)
        assert rewards.shape == (16,)
        assert next_states.shape == (16, total_states)
        assert dones.shape == (16,)


@skip_no_tf
class TestQNetwork:
    def test_output_shape(self):
        from deep_q_agent import build_q_network
        model = build_q_network(total_states, total_actions, [32, 32], 0.001)
        x = np.zeros((4, total_states), dtype=np.float32)
        out = model(x)
        assert out.shape == (4, total_actions)

    def test_model_compiles(self):
        from deep_q_agent import build_q_network
        model = build_q_network(total_states, total_actions, [64, 64], 0.001)
        assert model is not None


@skip_no_tf
class TestDQNAgent:
    def test_one_hot_shape(self):
        from deep_q_agent import DQNAgent
        vec = DQNAgent._one_hot_encode_state(5)
        assert vec.shape == (total_states,)
        assert vec[5] == 1.0
        assert vec.sum() == 1.0

    def test_select_action_feasible(self):
        from deep_q_agent import DQNAgent
        agent = DQNAgent(seed=0)
        agent.environment.reset()
        for _ in range(50):
            state = agent.environment.get_state()
            state_v = agent._one_hot_encode_state(state)
            possible = agent.environment.get_possible_actions()
            action = agent.select_action(state_v, possible)
            assert action in possible

    def test_learn_before_enough_samples(self):
        from deep_q_agent import DQNAgent
        agent = DQNAgent(seed=0)
        loss = agent.train_step()
        assert loss == 0.0

    def test_learn_after_filling_buffer(self):
        from deep_q_agent import DQNAgent
        agent = DQNAgent(seed=0)
        for _ in range(training_batch_size + 5):
            agent.replay_buffer.add_experience(np.zeros(total_states, dtype=np.float32),
                              0, 1.0,
                              np.zeros(total_states, dtype=np.float32),
                              False)
        loss = agent.train_step()
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_short_training_no_error(self):
        from deep_q_agent import DQNAgent
        import parameters as P
        orig = P.dqn_training_steps
        P.dqn_training_steps = 300
        P.logging_interval = 100
        agent = DQNAgent(seed=0)
        agent.train()
        P.dqn_training_steps = orig
        P.logging_interval = 10000

    def test_evaluate_returns_float(self):
        from deep_q_agent import DQNAgent
        agent = DQNAgent(seed=0)
        result = agent.evaluate_policy(num_evaluation_steps=100)
        assert isinstance(result, float)
        assert result >= 0.0
