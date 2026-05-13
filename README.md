# RL Project - Reinforcement Learning

A Python project for studying and implementing Reinforcement Learning algorithms.
Includes a Q-Learning agent that learns to escape a maze, with an interactive web visualization.

## Project Structure

```
rl-project/
├── agents/          # RL agent implementations (DQN, Q-Learning)
├── envs/            # Custom environments (Gymnasium compatible)
├── models/          # Neural network architectures (PyTorch)
├── utils/           # Helper functions (replay buffer, logging)
├── configs/         # Hyperparameter config files (YAML)
├── logs/            # Training logs & TensorBoard runs
├── checkpoints/     # Saved model weights & Q-tables
├── notebooks/       # Jupyter notebooks for experiments
├── tests/           # Unit tests
├── web/             # Web visualization (HTML/CSS/JS)
├── train.py         # Main training script
├── train_maze.py    # Maze Q-Learning training script
├── evaluate.py      # Evaluation script
└── requirements.txt # Dependencies
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Quick Start

### Train the maze agent

```bash
python train_maze.py
```

Options:

```bash
python train_maze.py --size 15 --episodes 1000 --lr 0.2
```

### Open web visualization

```bash
python -m http.server 8000 --directory web
```

Then open http://localhost:8000

### Train DQN on CartPole

```bash
python train.py --agent dqn --env CartPole-v1 --episodes 500
```

## Dependencies

- Python 3.10+
- PyTorch
- Gymnasium (OpenAI Gym successor)
- NumPy, Matplotlib
- TensorBoard
