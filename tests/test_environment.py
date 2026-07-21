

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from environment import Environment, encode_state, decode_state
from parameters  import total_states, data_queue_capacity, energy_queue_capacity, active_transmit_energy_per_packet, active_transmit_packets_per_slot, harvested_energy_per_power_level


class TestStateEncoding:
    def test_encode_decode_roundtrip(self):
        for jammer_state in range(2):
            for data_queue_level in range(0, data_queue_capacity + 1, 3):
                for energy_queue_level in range(0, energy_queue_capacity + 1, 3):
                    idx = encode_state(jammer_state, data_queue_level, energy_queue_level)
                    j2, d2, e2 = decode_state(idx)
                    assert (jammer_state, data_queue_level, energy_queue_level) == (j2, d2, e2)

    def test_index_range(self):
        for jammer_state in range(2):
            for data_queue_level in range(data_queue_capacity + 1):
                for energy_queue_level in range(energy_queue_capacity + 1):
                    idx = encode_state(jammer_state, data_queue_level, energy_queue_level)
                    assert 0 <= idx < total_states

    def test_unique_indices(self):
        seen = set()
        for jammer_state in range(2):
            for data_queue_level in range(data_queue_capacity + 1):
                for energy_queue_level in range(energy_queue_capacity + 1):
                    idx = encode_state(jammer_state, data_queue_level, energy_queue_level)
                    assert idx not in seen, f"Duplicate index {idx}"
                    seen.add(idx)
        assert len(seen) == total_states


class TestEnvironmentReset:
    def test_reset_returns_valid_index(self):
        env = Environment(seed=0)
        idx = env.reset()
        assert 0 <= idx < total_states

    def test_reset_zero_state(self):
        env = Environment(seed=0)
        env.reset()
        jammer_state, data_queue_level, energy_queue_level = env.state_tuple()
        assert jammer_state == 0 and data_queue_level == 0 and energy_queue_level == 0

    def test_get_state_matches_tuple(self):
        env = Environment(seed=7)
        env.reset()
        idx = env.get_state()
        jammer_state, data_queue_level, energy_queue_level = env.state_tuple()
        assert idx == encode_state(jammer_state, data_queue_level, energy_queue_level)


class TestPossibleActions:
    def test_idle_always_possible(self):
        env = Environment(seed=1)
        env.reset()
        for _ in range(50):
            env.perform_action(0)
            assert 0 in env.get_possible_actions()

    def test_active_tx_requires_jammer_idle(self):
        env = Environment(seed=0)
        env.reset()
        env.jammer_state = 1
        env.data_queue_level = 5
        env.energy_queue_level = 5
        possible = env.get_possible_actions()
        assert 1 not in possible

    def test_active_tx_requires_data(self):
        env = Environment(seed=0)
        env.reset()
        env.jammer_state = 0
        env.data_queue_level = 0
        env.energy_queue_level = 5
        possible = env.get_possible_actions()
        assert 1 not in possible

    def test_active_tx_requires_energy(self):
        env = Environment(seed=0)
        env.reset()
        env.jammer_state = 0
        env.data_queue_level = 5
        env.energy_queue_level = 0
        possible = env.get_possible_actions()
        assert 1 not in possible

    def test_backscatter_requires_jammer_active(self):
        env = Environment(seed=0)
        env.reset()
        env.jammer_state = 0
        env.data_queue_level = 5
        possible = env.get_possible_actions()
        assert 3 not in possible

    def test_harvest_requires_jammer_active(self):
        env = Environment(seed=0)
        env.reset()
        env.jammer_state = 0
        possible = env.get_possible_actions()
        assert 2 not in possible

    def test_all_actions_jammer_active_with_data(self):
        env = Environment(seed=0)
        env.reset()
        env.jammer_state = 1
        env.data_queue_level = 5
        env.energy_queue_level = 5
        possible = env.get_possible_actions()
        for a in [0, 2, 3, 4, 5, 6]:
            assert a in possible
        assert 1 not in possible


class TestReward:
    def test_idle_reward_zero(self):
        env = Environment(seed=0)
        env.reset()
        r, _ = env.perform_action(0)
        env2 = Environment(seed=5)
        env2.reset()
        env2.jammer_state = 0
        env2.data_queue_level = 5
        env2.energy_queue_level = 5
        r = env2._calculate_reward(0)
        assert r == 0

    def test_active_tx_reward_capped_by_data(self):
        env = Environment(seed=0)
        env.reset()
        env.jammer_state = 0
        env.data_queue_level = 2
        env.energy_queue_level = 10
        r = env._calculate_reward(1)
        assert r == 2

    def test_active_tx_fails_when_jammer_active(self):
        env = Environment(seed=0)
        env.reset()
        env.jammer_state = 1
        env.data_queue_level = 5
        env.energy_queue_level = 10
        r = env._calculate_reward(1)
        assert r == 0

    def test_harvest_reward_zero(self):
        env = Environment(seed=0)
        env.reset()
        env.jammer_state = 1
        r = env._calculate_reward(2)
        assert r == 0

    def test_backscatter_reward_nonneg(self):
        env = Environment(seed=42)
        env.reset()
        env.jammer_state = 1
        env.data_queue_level = 5
        r = env._calculate_reward(3)
        assert r >= 0


class TestQueueUpdates:
    def test_active_tx_consumes_data_and_energy(self):
        env = Environment(seed=0)
        env.reset()
        env.jammer_state = 0
        env.data_queue_level = 4
        env.energy_queue_level = 10
        env._calculate_reward(1)
        env._update_queues(1)
        assert env.data_queue_level == 0
        assert env.energy_queue_level == 10 - active_transmit_packets_per_slot * active_transmit_energy_per_packet

    def test_harvest_adds_energy(self):
        env = Environment(seed=0)
        env.reset()
        env.jammer_state = 1
        env.energy_queue_level = 0
        env.current_power_level = 0
        env._update_queues(2)
        assert env.energy_queue_level == 1

    def test_energy_capped_at_max(self):
        env = Environment(seed=0)
        env.reset()
        env.jammer_state = 1
        env.energy_queue_level = energy_queue_capacity
        env.current_power_level = 2
        env._update_queues(2)
        assert env.energy_queue_level == energy_queue_capacity

    def test_data_queue_capped_after_arrivals(self):
        env = Environment(seed=0)
        env.reset()
        env.data_queue_level = data_queue_capacity
        env._add_poisson_packet_arrivals()
        assert env.data_queue_level == data_queue_capacity


class TestJammerTransition:
    def test_jammer_transitions_are_valid(self):
        env = Environment(seed=13)
        env.reset()
        for _ in range(1000):
            env._transition_jammer_state()
            assert env.jammer_state in (0, 1)

    def test_jammer_average_active_rate(self):
        from parameters import jammer_idle_probability
        env = Environment(seed=99)
        env.reset()
        env.jammer_state = 1
        actives = 0
        N = 10000
        for _ in range(N):
            env._transition_jammer_state()
            actives += env.jammer_state
        assert abs(actives / N - (1 - jammer_idle_probability)) < 0.05, \
            f"Expected ~{1-jammer_idle_probability:.1f} active, got {actives/N:.3f}"
