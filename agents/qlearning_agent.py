"""
Tabular Q-Learning Agent for discrete state/action spaces.

Uses epsilon-greedy exploration with decay, and supports
JSON export/import of the Q-table for web visualization.
"""

import json
import random
import numpy as np
from collections import defaultdict


class QLearningAgent:
    """
    Q-Learning agent using a tabular Q-table.

    Suitable for environments with small discrete state and action spaces
    such as grid-based mazes.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        seed: int = 42,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q-table: state -> array of Q-values for each action
        self.q_table = defaultdict(lambda: np.zeros(action_size))

        # Stats
        self.total_updates = 0

        # Random seed
        self.rng = random.Random(seed)
        np.random.seed(seed)

    def select_action(self, state: int, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state (integer)
            training: If False, always exploit (greedy)

        Returns:
            Selected action index
        """
        if training and self.rng.random() < self.epsilon:
            return self.rng.randint(0, self.action_size - 1)
        else:
            q_values = self.q_table[state]
            # Break ties randomly
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return int(self.rng.choice(best_actions))

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool) -> float:
        """
        Update Q-value using the Q-Learning update rule:
        Q(s,a) = Q(s,a) + α * (r + γ * max_a' Q(s',a') - Q(s,a))

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended

        Returns:
            TD error (for logging)
        """
        current_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        td_error = target - current_q
        self.q_table[state][action] += self.lr * td_error

        self.total_updates += 1
        return abs(td_error)

    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def reset_epsilon(self):
        """Reset epsilon to initial value."""
        self.epsilon = self.epsilon_start

    def get_policy(self):
        """
        Get the greedy policy from the Q-table.

        Returns:
            Dictionary mapping state -> best action
        """
        policy = {}
        for state in self.q_table:
            q_values = self.q_table[state]
            policy[state] = int(np.argmax(q_values))
        return policy

    def save_qtable_json(self, filepath: str):
        """
        Save Q-table as JSON file (for web visualization).

        Format:
        {
            "state_size": int,
            "action_size": int,
            "epsilon": float,
            "total_updates": int,
            "q_table": { "state_id": [q_val_0, q_val_1, ...], ... }
        }
        """
        q_data = {}
        for state, q_values in self.q_table.items():
            q_data[str(state)] = q_values.tolist()

        data = {
            "state_size": self.state_size,
            "action_size": self.action_size,
            "epsilon": self.epsilon,
            "total_updates": self.total_updates,
            "q_table": q_data,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Q-table saved to {filepath} ({len(q_data)} states)")

    def load_qtable_json(self, filepath: str):
        """Load Q-table from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        self.state_size = data["state_size"]
        self.action_size = data["action_size"]
        self.epsilon = data["epsilon"]
        self.total_updates = data["total_updates"]

        self.q_table = defaultdict(lambda: np.zeros(self.action_size))
        for state_str, q_values in data["q_table"].items():
            self.q_table[int(state_str)] = np.array(q_values)

        print(f"Q-table loaded from {filepath} ({len(data['q_table'])} states)")

    def save(self, filepath: str):
        """Save agent state (numpy format for Python use)."""
        np.savez(
            filepath,
            q_table_keys=list(self.q_table.keys()),
            q_table_values=[self.q_table[k] for k in self.q_table],
            epsilon=self.epsilon,
            total_updates=self.total_updates,
            state_size=self.state_size,
            action_size=self.action_size,
        )

    def load(self, filepath: str):
        """Load agent state from numpy format."""
        data = np.load(filepath, allow_pickle=True)
        keys = data["q_table_keys"]
        values = data["q_table_values"]
        self.q_table = defaultdict(lambda: np.zeros(self.action_size))
        for k, v in zip(keys, values):
            self.q_table[int(k)] = v
        self.epsilon = float(data["epsilon"])
        self.total_updates = int(data["total_updates"])

    def __repr__(self):
        return (
            f"QLearningAgent(states={self.state_size}, actions={self.action_size}, "
            f"ε={self.epsilon:.4f}, updates={self.total_updates}, "
            f"q_entries={len(self.q_table)})"
        )
