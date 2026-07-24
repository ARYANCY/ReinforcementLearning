import argparse
import os

from q_learning_agent import QLearningAgent
from deep_q_agent import DQNAgent, TENSORFLOW_AVAILABLE
from generate_plots import generate_all_plots


def parse_command_line_arguments():
    parser = argparse.ArgumentParser(description="Train RL agents for Ambient Backscatter Anti-Jamming")
    parser.add_argument("--agent", choices=["q", "dqn", "both"], default="q", help="Which agent to train (default: q)")
    parser.add_argument("--eval", action="store_true", help="Run greedy evaluation after training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--plot", action="store_true", help="Generate all 5 plots after training")
    parser.add_argument("--steps", type=int, default=None, help="Custom training steps")
    return parser.parse_args()


def run_q_learning(args):
    agent = QLearningAgent(seed=args.seed)
    agent.train(steps=args.steps)
    if args.eval:
        agent.evaluate_policy()
    return agent


def run_dqn_agent(args):
    if not TENSORFLOW_AVAILABLE:
        print("[ERROR] TensorFlow is required to run DQN training.")
        return None
    agent = DQNAgent(seed=args.seed)
    agent.train(steps=args.steps)
    if args.eval:
        agent.evaluate_policy()
    return agent


if __name__ == "__main__":
    arguments = parse_command_line_arguments()
    if arguments.agent in ("q", "both"):
        run_q_learning(arguments)
    if arguments.agent in ("dqn", "both"):
        run_dqn_agent(arguments)
    if arguments.plot:
        generate_all_plots()
