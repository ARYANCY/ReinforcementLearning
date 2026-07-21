
import argparse


def parse_command_line_arguments():
    parser = argparse.ArgumentParser(description="Train RL agents for Ambient Backscatter Anti-Jamming")
    parser.add_argument("--agent", choices=["q", "dqn", "both"], default="q", help="Which agent to train (default: q)")
    parser.add_argument("--eval", action="store_true", help="Run greedy evaluation after training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--plot", action="store_true", help="Plot reward curves after training (requires matplotlib)")
    return parser.parse_args()


def run_q_learning(args):
    from q_learning_agent import QLearningAgent
    agent = QLearningAgent(seed=args.seed)
    agent.train()
    if args.eval:
        agent.evaluate_policy()
    return agent


def run_dqn_agent(args):
    from deep_q_agent import DQNAgent
    agent = DQNAgent(seed=args.seed)
    agent.train()
    if args.eval:
        agent.evaluate_policy()
    return agent


def plot_convergence_results():
    import os
    import csv
    import matplotlib.pyplot as plt
    from parameters import results_directory
    figure, axes = plt.subplots(figsize=(9, 5))
    for filename, label, color in [("q_learning_log.csv", "Q-Learning", "steelblue"), ("dqn_log.csv", "DQN", "darkorange")]:
        file_path = os.path.join(results_directory, filename)
        if not os.path.exists(file_path):
            continue
        steps, rewards = [], []
        with open(file_path) as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                steps.append(int(row["step"]))
                rewards.append(float(row["avg_reward"]))
        axes.plot(steps, rewards, label=label, color=color, linewidth=2)
    axes.set_xlabel("Training Steps", fontsize=12)
    axes.set_ylabel("Average Throughput (packets/slot)", fontsize=12)
    axes.set_title("Ambient Backscatter Anti-Jamming — RL Convergence", fontsize=13)
    axes.legend(fontsize=11)
    axes.grid(True, alpha=0.35)
    figure.tight_layout()
    output_file_path = os.path.join(results_directory, "convergence_plot.png")
    figure.savefig(output_file_path, dpi=150)
    print(f"  Plot saved -> {output_file_path}")
    plt.show()


if __name__ == "__main__":
    arguments = parse_command_line_arguments()
    if arguments.agent in ("q", "both"):
        run_q_learning(arguments)
    if arguments.agent in ("dqn", "both"):
        run_dqn_agent(arguments)
    if arguments.plot:
        plot_convergence_results()
