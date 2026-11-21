
import argparse
import random
from pathlib import Path
from statistics import mean

import numpy as np

from snake_env import SnakeEnvironment
from neural_network import NeuralNetwork


# ==========================================
# SNAKE AI - PHASE 5
# TRAINED MODEL EVALUATION
# ==========================================


def load_trained_model(model_path):
    """
    Load the weights saved by train.py.
    No learning takes place here.
    """

    if not Path(model_path).is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    network = NeuralNetwork(
        input_size=11,
        hidden_size=64,
        output_size=3
    )

    with np.load(
        model_path,
        allow_pickle=False
    ) as data:

        for name in ("W1", "b1", "W2", "b2"):

            saved = data[name]
            current = getattr(network, name)

            if saved.shape != current.shape:
                raise ValueError(
                    f"Incorrect shape for {name}"
                )

            np.copyto(current, saved)

        width = int(data["board_width"])
        height = int(data["board_height"])
        training_episodes = int(data["episodes"])

    return network, width, height, training_episodes


# ==========================================
# 1. PLAY ONE EVALUATION GAME
# ==========================================


def play_game(
    network,
    width,
    height,
    seed,
    use_random_actions=False
):

    env = SnakeEnvironment(
        width=width,
        height=height,
        seed=seed
    )

    rng = random.Random(seed + 100000)

    state = env.reset()
    done = False
    steps = 0

    while not done:

        if use_random_actions:

            action = rng.randrange(3)

        else:

            # Greedy policy: select highest Q-value
            q_values = network.predict(state)

            action = int(np.argmax(q_values))

        state, reward, done = env.step(action)

        steps += 1

    return env.score, steps


# ==========================================
# 2. EVALUATE MULTIPLE GAMES
# ==========================================


def evaluate(
    network,
    width,
    height,
    episodes
):

    ai_scores = []
    random_scores = []

    ai_steps = []
    random_steps = []

    for game in range(episodes):

        # Same starting seed for both policies
        game_seed = 1000 + game

        score, steps = play_game(
            network,
            width,
            height,
            seed=game_seed,
            use_random_actions=False
        )

        ai_scores.append(score)
        ai_steps.append(steps)

        score, steps = play_game(
            network,
            width,
            height,
            seed=game_seed,
            use_random_actions=True
        )

        random_scores.append(score)
        random_steps.append(steps)

    return {
        "ai_scores": ai_scores,
        "random_scores": random_scores,
        "ai_steps": ai_steps,
        "random_steps": random_steps
    }


# ==========================================
# 3. DISPLAY STATISTICS
# ==========================================


def print_summary(title, scores, steps):

    print(f"\n{title}")
    print("-" * 40)

    print(f"Games played: {len(scores)}")
    print(f"Average score: {mean(scores):.3f}")
    print(f"Best score: {max(scores)}")

    print(
        f"Average movements: {mean(steps):.2f}"
    )

    zero_games = sum(
        score == 0 for score in scores
    )

    zero_percentage = (
        100 * zero_games / len(scores)
    )

    print(
        f"Games with zero food: "
        f"{zero_percentage:.1f}%"
    )


# ==========================================
# 4. MAIN PROGRAM
# ==========================================


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate a trained Snake AI"
    )

    parser.add_argument(
        "--model",
        default="artifacts/smoke/snake_model.npz"
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=50
    )

    args = parser.parse_args()

    if args.episodes < 1:
        raise ValueError(
            "Episodes must be positive."
        )

    network, width, height, trained_games = (
        load_trained_model(args.model)
    )

    print("\nSNAKE AI - MODEL EVALUATION")
    print("=" * 45)

    print(f"Model: {args.model}")
    print(f"Board: {width} x {height}")
    print(f"Training episodes: {trained_games}")
    print(f"Evaluation episodes: {args.episodes}")
    print("AI exploration: Disabled")

    results = evaluate(
        network,
        width,
        height,
        args.episodes
    )

    print_summary(
        "TRAINED AI",
        results["ai_scores"],
        results["ai_steps"]
    )

    print_summary(
        "RANDOM PLAYER",
        results["random_scores"],
        results["random_steps"]
    )

    ai_average = mean(results["ai_scores"])
    random_average = mean(results["random_scores"])

    print("\n" + "=" * 45)
    print("COMPARISON")
    print("=" * 45)

    print(f"AI average: {ai_average:.3f}")
    print(f"Random average: {random_average:.3f}")

    if ai_average > random_average:
        print("AI outperformed random in this sample.")

    elif ai_average < random_average:
        print("Random outperformed AI in this sample.")

    else:
        print("Both policies achieved equal averages.")

    print("\nEvaluation completed.")
    print("No model weights were modified.")


if __name__ == "__main__":
    main()
