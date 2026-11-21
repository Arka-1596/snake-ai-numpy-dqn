
# Snake AI — Deep Q-Learning from Scratch

A reinforcement learning project where a neural network learns
to play Snake using NumPy, without TensorFlow or PyTorch.

## Overview

The AI observes an 11-feature representation of the game
environment and predicts Q-values for three possible actions:

- Move straight
- Turn right
- Turn left

The project implements forward propagation, backpropagation,
Deep Q-Learning, experience replay, epsilon-greedy exploration,
and a target network.

## Technology Stack

- Python
- NumPy
- Pygame
- Matplotlib

## Neural Network Architecture

11 input neurons → 64 hidden neurons → 3 output neurons

The hidden layer uses ReLU activation. The output layer
produces Q-value predictions.

## Evaluation Results

Initial model trained for 100 episodes on a 12 × 12 board.

| Metric | Trained AI | Random Player |
|---|---:|---:|
| Average score | 5.64 | 0.14 |
| Best score | 24 | 1 |
| Evaluation games | 50 | 50 |
| Games with zero food | 8% | 86% |

These are results from one evaluation sample, not
guaranteed performance across all random seeds.

## Features

- Snake game environment
- 11-dimensional state representation
- Neural network implemented from scratch
- Manual backpropagation
- Deep Q-Learning agent
- Experience replay
- Target network
- Training and evaluation scripts
- Live neural-network visualization
- Training performance dashboard

## Installation

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running the Project

Play Snake manually:

```bash
python snake_game.py
```

Run the environment tests:

```bash
python test_env.py
```

Train a model:

```bash
python train.py --games 100
```

Evaluate the trained model:

```bash
python evaluate_model.py --model artifacts/smoke/snake_model.npz
```

Watch the AI play:

```bash
python play_ai_visual.py --model artifacts/smoke/snake_model.npz
```

Generate training charts:

```bash
python plot_training.py
```

## Known Limitations

The current agent can enter repetitive movement loops.
Reward shaping and further evaluation are being developed
to improve food-seeking behaviour.

## Future Work

- Improved reward strategies
- Loop detection and state representation
- Evaluation across multiple random seeds
- Browser-based AI visualization
- Interactive website deployed on Vercel

## Project Status

The NumPy DQN implementation, Pygame environment,
model evaluation, and live visualizer are working.

Model optimization and the browser showcase are in progress.
