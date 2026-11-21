
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ==========================================
# SNAKE AI - PHASE 6
# TRAINING PERFORMANCE DASHBOARD
# ==========================================


def load_history(csv_path):

    path = Path(csv_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Training history not found: {csv_path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("Training history is empty.")

    episodes = []
    scores = []
    averages = []
    epsilons = []
    losses = []

    for row in rows:

        episodes.append(int(row["episode"]))
        scores.append(int(row["score"]))

        averages.append(
            float(row["average_score_20"])
        )

        epsilons.append(
            float(row["epsilon"])
        )

        loss_value = row["average_loss"].strip()

        losses.append(
            float(loss_value)
            if loss_value
            else np.nan
        )

    return {
        "episodes": np.array(episodes),
        "scores": np.array(scores),
        "averages": np.array(averages),
        "epsilons": np.array(epsilons),
        "losses": np.array(losses)
    }


def generate_dashboard(history, output_path):

    episodes = history["episodes"]
    scores = history["scores"]
    averages = history["averages"]
    epsilons = history["epsilons"]
    losses = history["losses"]

    plt.style.use("dark_background")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 8)
    )

    fig.suptitle(
        "SNAKE AI | TRAINING ANALYSIS",
        fontsize=19,
        fontweight="bold"
    )

    # ======================================
    # GRAPH 1: SCORE PER GAME
    # ======================================

    ax = axes[0, 0]

    ax.plot(
        episodes,
        scores,
        color="#50BFAE",
        linewidth=1,
        alpha=0.7
    )

    ax.scatter(
        episodes,
        scores,
        s=9,
        color="#81E5D4"
    )

    ax.set_title("Score per game")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Food collected")
    ax.grid(alpha=0.15)

    # ======================================
    # GRAPH 2: MOVING AVERAGE
    # ======================================

    ax = axes[0, 1]

    ax.plot(
        episodes,
        averages,
        color="#E8B66E",
        linewidth=2
    )

    ax.set_title("Average score (last 20 games)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average score")
    ax.grid(alpha=0.15)

    # ======================================
    # GRAPH 3: EPSILON DECAY
    # ======================================

    ax = axes[1, 0]

    ax.plot(
        episodes,
        epsilons,
        color="#A5A4EF",
        linewidth=2
    )

    ax.set_title("Exploration probability")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.15)

    # ======================================
    # GRAPH 4: TRAINING LOSS
    # ======================================

    ax = axes[1, 1]

    ax.plot(
        episodes,
        losses,
        color="#F28F91",
        linewidth=1.2
    )

    ax.set_title("Average training loss per game")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.15)

    # ======================================
    # SAVE DASHBOARD
    # ======================================

    fig.tight_layout(
        rect=(0, 0, 1, 0.95)
    )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nTRAINING DASHBOARD GENERATED")
    print("=" * 40)

    print(f"Episodes: {len(episodes)}")
    print(f"Best score: {max(scores)}")
    print(f"Final average: {averages[-1]:.3f}")
    print(f"Final epsilon: {epsilons[-1]:.4f}")

    print(f"\nDashboard saved: {output_path}")


def main():

    parser = argparse.ArgumentParser(
        description="Plot Snake AI training statistics"
    )

    parser.add_argument(
        "--csv",
        default="artifacts/smoke/training_history.csv"
    )

    parser.add_argument(
        "--output",
        default="artifacts/smoke/training_dashboard.png"
    )

    args = parser.parse_args()

    history = load_history(args.csv)

    generate_dashboard(
        history,
        args.output
    )


if __name__ == "__main__":
    main()
