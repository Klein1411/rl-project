"""
Custom Gymnasium Maze Environment.

A grid-based maze where an RL agent must find the exit.
The maze is generated using DFS recursive backtracker algorithm,
guaranteeing a perfect maze (exactly one path between any two cells).
"""

import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MazeEnv(gym.Env):
    """
    Maze environment for Reinforcement Learning.

    Observation:
        Agent's position encoded as a single integer: row * width + col

    Actions:
        0: UP
        1: RIGHT
        2: DOWN
        3: LEFT

    Rewards:
        +100  for reaching the goal
        -1    for each step (encourages shortest path)
        -5    for hitting a wall (discourage wall bumps)
    """

    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 10}

    # Action constants
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    ACTION_NAMES = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
    ACTION_DELTAS = {
        0: (-1, 0),   # UP
        1: (0, 1),    # RIGHT
        2: (1, 0),    # DOWN
        3: (0, -1),   # LEFT
    }

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        render_mode: str = None,
        seed: int = None,
        reward_goal: float = 100.0,
        reward_step: float = -1.0,
        reward_wall: float = -5.0,
        random_start: bool = False,
        random_goal: bool = False,
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.render_mode = render_mode
        self.reward_goal = reward_goal
        self.reward_step = reward_step
        self.reward_wall = reward_wall
        self.random_start = random_start
        self.random_goal = random_goal

        # State and action spaces
        self.observation_space = spaces.Discrete(width * height)
        self.action_space = spaces.Discrete(4)

        # Max steps to prevent infinite loops
        self.max_steps = width * height * 4

        # Generate the maze
        self._rng = random.Random(seed)
        self.maze = None  # Will be set in _generate_maze
        self.walls = None  # Set of wall pairs between cells
        self.start_pos = None
        self.goal_pos = None
        self.agent_pos = None
        self.steps_taken = 0
        self.visited = set()

        self._generate_maze()

    def _generate_maze(self):
        """
        Generate a perfect maze using DFS recursive backtracker.

        The maze is represented as a set of walls between adjacent cells.
        Initially all walls exist; the algorithm carves passages by removing walls.
        """
        w, h = self.width, self.height

        # Track which cells have been visited during generation
        visited = set()
        # Set of walls: each wall is a frozenset of two adjacent cell coords
        self.walls = set()

        # Initialize all walls
        for r in range(h):
            for c in range(w):
                if r > 0:
                    self.walls.add(frozenset(((r, c), (r - 1, c))))
                if c > 0:
                    self.walls.add(frozenset(((r, c), (r, c - 1))))

        # DFS maze generation (iterative to avoid stack overflow)
        stack = [(0, 0)]
        visited.add((0, 0))

        while stack:
            current = stack[-1]
            r, c = current

            # Find unvisited neighbors
            neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                    neighbors.append((nr, nc))

            if neighbors:
                # Choose random unvisited neighbor
                next_cell = self._rng.choice(neighbors)
                # Remove wall between current and next
                wall = frozenset((current, next_cell))
                self.walls.discard(wall)
                visited.add(next_cell)
                stack.append(next_cell)
            else:
                stack.pop()

        # Set start and goal positions
        if self.random_start:
            self.start_pos = (self._rng.randint(0, h - 1), self._rng.randint(0, w - 1))
        else:
            self.start_pos = (0, 0)

        if self.random_goal:
            self.goal_pos = (self._rng.randint(0, h - 1), self._rng.randint(0, w - 1))
            while self.goal_pos == self.start_pos:
                self.goal_pos = (self._rng.randint(0, h - 1), self._rng.randint(0, w - 1))
        else:
            self.goal_pos = (h - 1, w - 1)

        # Build adjacency info for quick passage checks
        self._passages = set()
        for r in range(h):
            for c in range(w):
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        wall = frozenset(((r, c), (nr, nc)))
                        if wall not in self.walls:
                            self._passages.add(((r, c), (nr, nc)))

    def _can_move(self, from_pos, to_pos):
        """Check if there is a passage (no wall) between two cells."""
        r, c = to_pos
        if not (0 <= r < self.height and 0 <= c < self.width):
            return False
        return ((from_pos, to_pos) in self._passages or
                (to_pos, from_pos) in self._passages)

    def _pos_to_state(self, pos):
        """Convert (row, col) to integer state."""
        return pos[0] * self.width + pos[1]

    def _state_to_pos(self, state):
        """Convert integer state to (row, col)."""
        return (state // self.width, state % self.width)

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        self.agent_pos = self.start_pos
        self.steps_taken = 0
        self.visited = {self.start_pos}

        observation = self._pos_to_state(self.agent_pos)
        info = {
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "steps": self.steps_taken,
        }
        return observation, info

    def step(self, action):
        """Execute one step in the environment."""
        assert self.action_space.contains(action), f"Invalid action: {action}"

        dr, dc = self.ACTION_DELTAS[action]
        new_pos = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)

        self.steps_taken += 1
        hit_wall = False

        if self._can_move(self.agent_pos, new_pos):
            self.agent_pos = new_pos
            self.visited.add(new_pos)
        else:
            hit_wall = True

        # Calculate reward
        if self.agent_pos == self.goal_pos:
            reward = self.reward_goal
            terminated = True
        elif hit_wall:
            reward = self.reward_wall
            terminated = False
        else:
            reward = self.reward_step
            terminated = False

        # Check if max steps exceeded
        truncated = self.steps_taken >= self.max_steps

        observation = self._pos_to_state(self.agent_pos)
        info = {
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "steps": self.steps_taken,
            "hit_wall": hit_wall,
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the maze."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "rgb_array":
            return self._render_rgb()
        return None

    def _render_ansi(self):
        """Render the maze as ASCII art."""
        h, w = self.height, self.width
        # Create a character grid
        # Each cell is 2 chars wide, 1 char tall
        # Plus borders
        lines = []

        # Top border
        top = "+"
        for c in range(w):
            top += "---+"
        lines.append(top)

        for r in range(h):
            # Cell row
            row = "|"
            for c in range(w):
                if (r, c) == self.agent_pos:
                    cell = " A "
                elif (r, c) == self.goal_pos:
                    cell = " G "
                elif (r, c) == self.start_pos:
                    cell = " S "
                elif (r, c) in self.visited:
                    cell = " . "
                else:
                    cell = "   "
                # Right wall
                if c < w - 1:
                    wall = frozenset(((r, c), (r, c + 1)))
                    right = "|" if wall in self.walls else " "
                else:
                    right = "|"
                row += cell + right
            lines.append(row)

            # Bottom wall row
            bottom = "+"
            for c in range(w):
                if r < h - 1:
                    wall = frozenset(((r, c), (r + 1, c)))
                    bwall = "---" if wall in self.walls else "   "
                else:
                    bwall = "---"
                bottom += bwall + "+"
            lines.append(bottom)

        result = "\n".join(lines)
        print(result)
        return result

    def _render_rgb(self):
        """Render the maze as an RGB array."""
        cell_size = 20
        h, w = self.height, self.width
        img_h = h * cell_size + 1
        img_w = w * cell_size + 1
        img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255

        # Draw walls
        for r in range(h):
            for c in range(w):
                x = c * cell_size
                y = r * cell_size

                # Right wall
                if c < w - 1:
                    wall = frozenset(((r, c), (r, c + 1)))
                    if wall in self.walls:
                        img[y:y + cell_size + 1, x + cell_size, :] = 0

                # Bottom wall
                if r < h - 1:
                    wall = frozenset(((r, c), (r + 1, c)))
                    if wall in self.walls:
                        img[y + cell_size, x:x + cell_size + 1, :] = 0

        # Draw borders
        img[0, :, :] = 0
        img[-1, :, :] = 0
        img[:, 0, :] = 0
        img[:, -1, :] = 0

        # Draw start (green)
        sr, sc = self.start_pos
        sx, sy = sc * cell_size + 2, sr * cell_size + 2
        img[sy:sy + cell_size - 3, sx:sx + cell_size - 3] = [0, 200, 0]

        # Draw goal (red)
        gr, gc = self.goal_pos
        gx, gy = gc * cell_size + 2, gr * cell_size + 2
        img[gy:gy + cell_size - 3, gx:gx + cell_size - 3] = [200, 0, 0]

        # Draw agent (blue)
        ar, ac = self.agent_pos
        ax, ay = ac * cell_size + 3, ar * cell_size + 3
        img[ay:ay + cell_size - 5, ax:ax + cell_size - 5] = [0, 0, 255]

        return img

    def get_maze_data(self):
        """
        Export maze data as a dictionary (for JSON serialization).
        Returns wall information as list of cell pairs.
        """
        walls_list = []
        for wall in self.walls:
            cells = list(wall)
            walls_list.append([list(cells[0]), list(cells[1])])

        return {
            "width": self.width,
            "height": self.height,
            "walls": walls_list,
            "start": list(self.start_pos),
            "goal": list(self.goal_pos),
        }
