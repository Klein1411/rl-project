import argparse
import gymnasium as gym
import torch
import yaml
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train an RL agent")
    parser.add_argument("--agent", type=str, default="dqn", choices=["dqn", "ppo", "a2c"],
                        help="RL algorithm to use")
    parser.add_argument("--env", type=str, default="CartPole-v1",
                        help="Gymnasium environment ID")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of training episodes")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (overrides CLI args)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cpu', 'cuda', or 'auto'")
    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def main():
    args = parse_args()

    # Load config from YAML if provided
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            setattr(args, key, value)

    device = get_device(args.device)
    print(f"Using device: {device}")
    print(f"Training {args.agent.upper()} on {args.env} for {args.episodes} episodes")

    # Set random seeds
    torch.manual_seed(args.seed)

    # Create environment
    env = gym.make(args.env)

    # TODO: Initialize agent based on args.agent
    # TODO: Training loop
    # TODO: Save checkpoints to checkpoints/
    # TODO: Log metrics to logs/

    print("Training complete!")
    env.close()


if __name__ == "__main__":
    main()
