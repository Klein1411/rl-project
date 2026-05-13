"""
Training script for Q-Learning agent on the Maze environment.

Trains the agent and exports:
  - Q-table (JSON) for web visualization
  - Maze layout (JSON) for web visualization
  - Training history (JSON) for analysis
  - Agent checkpoint (npz) for Python evaluation

Usage:
    python train_maze.py
    python train_maze.py --size 15 --episodes 1000
    python train_maze.py --size 10 --episodes 500 --lr 0.2
"""

import argparse
import json
import os
import time

import numpy as np
from tqdm import tqdm

from envs.maze_env import MazeEnv
from agents.qlearning_agent import QLearningAgent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Q-Learning agent to solve a maze"
    )
    parser.add_argument(
        "--size", type=int, default=10,
        help="Maze size (width and height, default: 10)"
    )
    parser.add_argument(
        "--width", type=int, default=None,
        help="Maze width (overrides --size)"
    )
    parser.add_argument(
        "--height", type=int, default=None,
        help="Maze height (overrides --size)"
    )
    parser.add_argument(
        "--episodes", type=int, default=500,
        help="Number of training episodes (default: 500)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.1,
        help="Learning rate (default: 0.1)"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99,
        help="Discount factor (default: 0.99)"
    )
    parser.add_argument(
        "--epsilon-start", type=float, default=1.0,
        help="Initial epsilon for exploration (default: 1.0)"
    )
    parser.add_argument(
        "--epsilon-end", type=float, default=0.01,
        help="Final epsilon (default: 0.01)"
    )
    parser.add_argument(
        "--epsilon-decay", type=float, default=0.995,
        help="Epsilon decay rate per episode (default: 0.995)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--render-interval", type=int, default=0,
        help="Render maze every N episodes (0=disabled, default: 0)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="checkpoints",
        help="Output directory for saved files (default: checkpoints)"
    )
    parser.add_argument(
        "--web-dir", type=str, default="web",
        help="Web directory for JSON exports (default: web)"
    )
    return parser.parse_args()


def train(args):
    """Main training function."""
    width = args.width or args.size
    height = args.height or args.size

    print("=" * 60)
    print("  MAZE REINFORCEMENT LEARNING TRAINER")
    print("=" * 60)
    print(f"  Maze Size    : {width} × {height}")
    print(f"  Episodes     : {args.episodes}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Gamma        : {args.gamma}")
    print(f"  Epsilon      : {args.epsilon_start} → {args.epsilon_end} (decay={args.epsilon_decay})")
    print(f"  Seed         : {args.seed}")
    print("=" * 60)

    # Create environment
    env = MazeEnv(
        width=width,
        height=height,
        render_mode="ansi" if args.render_interval > 0 else None,
        seed=args.seed,
    )

    # Create agent
    agent = QLearningAgent(
        state_size=width * height,
        action_size=4,
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
    )

    print(f"\n  Agent: {agent}")
    print(f"  Start: {env.start_pos}  →  Goal: {env.goal_pos}")
    print()

    # Training history
    history = {
        "episode_rewards": [],
        "episode_steps": [],
        "episode_epsilon": [],
        "episode_solved": [],
        "td_errors": [],
    }

    # Training loop
    best_reward = -float("inf")
    solved_count = 0
    start_time = time.time()

    pbar = tqdm(range(1, args.episodes + 1), desc="Training", unit="ep")
    for episode in pbar:
        state, info = env.reset()
        total_reward = 0.0
        total_td_error = 0.0
        steps = 0
        done = False

        while not done:
            # Select action
            action = agent.select_action(state, training=True)

            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Update Q-table
            td_error = agent.update(state, action, reward, next_state, terminated)
            total_td_error += td_error

            state = next_state
            total_reward += reward
            steps += 1

        # Decay epsilon
        agent.decay_epsilon()

        # Track if solved
        solved = terminated  # reached goal (not truncated)
        if solved:
            solved_count += 1

        # Record history
        history["episode_rewards"].append(total_reward)
        history["episode_steps"].append(steps)
        history["episode_epsilon"].append(agent.epsilon)
        history["episode_solved"].append(solved)
        history["td_errors"].append(total_td_error / max(steps, 1))

        # Update best reward
        if total_reward > best_reward:
            best_reward = total_reward

        # Update progress bar
        recent_rewards = history["episode_rewards"][-50:]
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        recent_solved = history["episode_solved"][-50:]
        solve_rate = sum(recent_solved) / len(recent_solved) * 100

        pbar.set_postfix({
            "R": f"{total_reward:.0f}",
            "avg50": f"{avg_reward:.1f}",
            "steps": steps,
            "ε": f"{agent.epsilon:.3f}",
            "solve%": f"{solve_rate:.0f}%",
        })

        # Render if requested
        if args.render_interval > 0 and episode % args.render_interval == 0:
            print(f"\n--- Episode {episode} ---")
            env.render()

    elapsed = time.time() - start_time

    # Training summary
    print("\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Total Time        : {elapsed:.1f}s")
    print(f"  Total Updates     : {agent.total_updates:,}")
    print(f"  Final Epsilon     : {agent.epsilon:.4f}")
    print(f"  Best Reward       : {best_reward:.1f}")
    print(f"  Q-table Entries   : {len(agent.q_table)}")

    # Last 100 episodes stats
    last_n = min(100, args.episodes)
    recent_rewards = history["episode_rewards"][-last_n:]
    recent_steps = history["episode_steps"][-last_n:]
    recent_solved = history["episode_solved"][-last_n:]

    print(f"\n  Last {last_n} Episodes:")
    print(f"    Avg Reward      : {np.mean(recent_rewards):.1f}")
    print(f"    Avg Steps       : {np.mean(recent_steps):.1f}")
    print(f"    Solve Rate      : {sum(recent_solved)/len(recent_solved)*100:.1f}%")
    print(f"    Min Steps       : {min(recent_steps)}")
    print("=" * 60)

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.web_dir, exist_ok=True)

    # 1. Save Q-table (JSON for web)
    qtable_path = os.path.join(args.output_dir, "maze_qtable.json")
    agent.save_qtable_json(qtable_path)

    # Also copy to web dir
    web_qtable_path = os.path.join(args.web_dir, "maze_qtable.json")
    agent.save_qtable_json(web_qtable_path)

    # 2. Save maze layout (JSON for web)
    maze_data = env.get_maze_data()
    maze_path = os.path.join(args.output_dir, "maze_layout.json")
    with open(maze_path, "w") as f:
        json.dump(maze_data, f, indent=2)
    print(f"Maze layout saved to {maze_path}")

    web_maze_path = os.path.join(args.web_dir, "maze_layout.json")
    with open(web_maze_path, "w") as f:
        json.dump(maze_data, f, indent=2)
    print(f"Maze layout saved to {web_maze_path}")

    # 3. Save training history
    history_path = os.path.join(args.output_dir, "training_history.json")
    # Convert booleans for JSON
    history["episode_solved"] = [bool(s) for s in history["episode_solved"]]
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")

    web_history_path = os.path.join(args.web_dir, "training_history.json")
    with open(web_history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {web_history_path}")

    # 4. Save agent checkpoint
    checkpoint_path = os.path.join(args.output_dir, "maze_agent.npz")
    agent.save(checkpoint_path)
    print(f"Agent checkpoint saved to {checkpoint_path}")

    # 5. Demonstrate solution
    print("\n" + "=" * 60)
    print("  DEMONSTRATING TRAINED AGENT")
    print("=" * 60)

    demo_env = MazeEnv(
        width=width, height=height,
        render_mode="ansi", seed=args.seed,
    )
    state, _ = demo_env.reset()
    done = False
    demo_steps = 0
    demo_reward = 0

    path = [demo_env.agent_pos]
    while not done and demo_steps < width * height * 2:
        action = agent.select_action(state, training=False)
        state, reward, terminated, truncated, info = demo_env.step(action)
        done = terminated or truncated
        demo_steps += 1
        demo_reward += reward
        path.append(demo_env.agent_pos)

    demo_env.render()
    print(f"\n  Solution: {demo_steps} steps, reward: {demo_reward:.1f}")
    print(f"  {'SOLVED!' if terminated else 'Failed to solve'}")

    # Save solution path for web
    solution_data = {
        "path": [list(p) for p in path],
        "steps": demo_steps,
        "reward": demo_reward,
        "solved": bool(terminated),
    }
    solution_path = os.path.join(args.web_dir, "solution.json")
    with open(solution_path, "w") as f:
        json.dump(solution_data, f, indent=2)
    print(f"Solution path saved to {solution_path}")

    env.close()
    demo_env.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)
