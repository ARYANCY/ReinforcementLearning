
import numpy as np
from parameters import (
    jammer_idle_probability,
    jammer_power_probabilities,
    packet_arrival_rate,
    data_queue_capacity,
    energy_queue_capacity,
    active_transmit_packets_per_slot,
    active_transmit_energy_per_packet,
    backscatter_packets_delivered,
    harvested_energy_per_power_level,
    rate_adaptation_packets_delivered,
    fixed_backscatter_rate,
    total_actions,
    total_states,
    num_rate_adaptation_levels,
)


def encode_state(jammer_state: int, data_queue_level: int, energy_queue_level: int) -> int:
    return jammer_state * (data_queue_capacity + 1) * (energy_queue_capacity + 1) + data_queue_level * (energy_queue_capacity + 1) + energy_queue_level


def decode_state(state_index: int):
    energy_queue_level = state_index % (energy_queue_capacity + 1)
    state_index //= (energy_queue_capacity + 1)
    data_queue_level = state_index % (data_queue_capacity + 1)
    jammer_state = state_index // (data_queue_capacity + 1)
    return jammer_state, data_queue_level, energy_queue_level


class Environment:
    def __init__(self, seed: int = None):
        self.random_generator = np.random.default_rng(seed)
        self.jammer_state = 0
        self.data_queue_level = 0
        self.energy_queue_level = 0
        self.current_power_level = None

    def reset(self):
        self.jammer_state = 0
        self.data_queue_level = 0
        self.energy_queue_level = 0
        self.current_power_level = None
        return encode_state(self.jammer_state, self.data_queue_level, self.energy_queue_level)

    def get_state(self) -> int:
        return encode_state(self.jammer_state, self.data_queue_level, self.energy_queue_level)

    def get_possible_actions(self) -> list:
        possible_actions = [0]
        if self.jammer_state == 0 and self.data_queue_level > 0 and self.energy_queue_level >= active_transmit_energy_per_packet:
            possible_actions.append(1)
        if self.jammer_state == 1:
            possible_actions.append(2)
            if self.data_queue_level > 0:
                possible_actions.append(3)
                for adaptation_level in range(num_rate_adaptation_levels):
                    possible_actions.append(4 + adaptation_level)
        return possible_actions

    def perform_action(self, action: int):
        reward = self._calculate_reward(action)
        self._update_queues(action)
        self._transition_jammer_state()
        self._add_poisson_packet_arrivals()
        return reward, self.get_state()

    def _sample_jammer_power_level(self) -> int:
        return int(self.random_generator.choice(len(jammer_power_probabilities), p=jammer_power_probabilities))

    def _calculate_reward(self, action: int) -> int:
        if action == 0:
            return 0
        if action == 1:
            if self.jammer_state == 1:
                return 0
            packets_sent = min(active_transmit_packets_per_slot, self.data_queue_level)
            return packets_sent
        power_level = self._sample_jammer_power_level()
        self.current_power_level = power_level
        if action == 2:
            return 0
        if action == 3:
            if self.jammer_state == 0:
                return 0
            packets_sent = min(backscatter_packets_delivered[power_level], self.data_queue_level)
            return packets_sent
        if 4 <= action <= 3 + num_rate_adaptation_levels:
            if self.jammer_state == 0:
                return 0
            adaptation_level = action - 4
            packets_sent = min(rate_adaptation_packets_delivered[adaptation_level], self.data_queue_level)
            return packets_sent
        return 0

    def _update_queues(self, action: int):
        power_level = self.current_power_level
        if action == 1 and self.jammer_state == 0:
            packets_sent = min(active_transmit_packets_per_slot, self.data_queue_level)
            self.data_queue_level -= packets_sent
            self.energy_queue_level -= min(packets_sent * active_transmit_energy_per_packet, self.energy_queue_level)
        elif action == 2 and self.jammer_state == 1 and power_level is not None:
            energy_gained = harvested_energy_per_power_level[power_level]
            self.energy_queue_level = min(self.energy_queue_level + energy_gained, energy_queue_capacity)
        elif action == 3 and self.jammer_state == 1 and power_level is not None:
            packets_sent = min(backscatter_packets_delivered[power_level], self.data_queue_level)
            self.data_queue_level -= packets_sent
        elif 4 <= action <= 3 + num_rate_adaptation_levels and self.jammer_state == 1 and power_level is not None:
            adaptation_level = action - 4
            packets_sent = min(rate_adaptation_packets_delivered[adaptation_level], self.data_queue_level)
            self.data_queue_level -= packets_sent
        self.current_power_level = None

    def _transition_jammer_state(self):
        self.jammer_state = 0 if self.random_generator.random() < jammer_idle_probability else 1

    def _add_poisson_packet_arrivals(self):
        packet_arrivals = self.random_generator.poisson(packet_arrival_rate)
        self.data_queue_level = min(self.data_queue_level + packet_arrivals, data_queue_capacity)

    def state_tuple(self):
        return (self.jammer_state, self.data_queue_level, self.energy_queue_level)

    def __repr__(self):
        return f"Environment(jammer_state={self.jammer_state}, data_queue_level={self.data_queue_level}, energy_queue_level={self.energy_queue_level}, state_index={self.get_state()})"
