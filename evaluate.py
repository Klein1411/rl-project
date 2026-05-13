import argparse
import gymnasium as gym
import torch
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained RL agent")
    parser.add_argument("--agent", type=str, default="dqn", choices=["dqn", "ppo", "a2c"])
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to saved model checkpoint")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes") 
    parser.add_argument("--render", action="store_true",
                        help="Render the environment")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()

def main():
    args = parse_args()

    render_mode = "human" if args.render else None
    env = gym.make(args.env, render_mode=render_mode)

    # TODO: Load agent from checkpoint
    # TODO: Run evaluation episodes
    # TODO: Print average reward

    print(f"Evaluating {args.agent.upper()} on {args.env}")
    print(f"Checkpoint: {args.checkpoint}")

    env.close()


if __name__ == "__main__":
    main()
