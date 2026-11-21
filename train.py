import argparse
import csv
from pathlib import Path

import numpy as np

from snake_env import SnakeEnvironment
from dqn_agent import DQNAgent


# ==========================================
# SNAKE AI - PHASE 5
# DEEP Q-LEARNING TRAINING PROGRAM
# ==========================================


def save_model(agent, path, width, height, games):
    """
    Save the trained neural network weights.

    This saves a model for inference.
    It does not save replay memory or
    the complete training state.
    """

    network = agent.policy_network

    np.savez_compressed(
        path,
        W1=network.W1,
        b1=network.b1,
        W2=network.W2,
        b2=network.b2,
        board_width=width,
        board_height=height,
        episodes=games
    )


def save_history(history, path):
    """
    Save training statistics to a CSV file.
    """

    columns = [
        "episode",
        "score",
        "average_score_20",
        "best_score",
        "steps",
        "epsilon",
        "total_reward",
        "average_loss"
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns
        )

        writer.writeheader()
        writer.writerows(history)


def run_training(
    games=50,
    width=12,
    height=12,
    batch_size=16,
    train_every=4,
    log_every=10,
    seed=42,
    output_dir="artifacts"
):
    """
    Train the Snake AI using Deep Q-Learning.
    """

    if games < 1:
        raise ValueError("Games must be positive.")

    if train_every < 1:
        raise ValueError(
            "Training interval must be positive."
        )

    if log_every < 1:
        raise ValueError(
            "Logging interval must be positive."
        )

    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    # ======================================
    # 1. INITIALIZE ENVIRONMENT
    # ======================================

    env = SnakeEnvironment(
        width=width,
        height=height,
        seed=seed
    )

    # ======================================
    # 2. INITIALIZE AI AGENT
    # ======================================

    agent = DQNAgent(
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        learning_rate=0.001,
        memory_size=50000,
        batch_size=batch_size,
        target_sync_interval=200,
        seed=seed
    )

    history = []

    best_score = 0
    total_steps = 0

    print("\nSNAKE AI - PHASE 5 TRAINING")
    print("=" * 55)

    print(f"Training episodes: {games}")
    print(f"Board size: {width} x {height}")
    print(f"Replay batch size: {batch_size}")
    print(f"Training interval: {train_every}")
    print(f"Initial epsilon: {agent.epsilon:.3f}")

    print("=" * 55)

    # ======================================
    # 3. MAIN TRAINING LOOP
    # ======================================

    for episode in range(1, games + 1):

        # Reset game
        state = env.reset()

        done = False

        episode_steps = 0
        episode_reward = 0.0
        episode_losses = []

        # ==================================
        # 4. PLAY ONE COMPLETE GAME
        # ==================================

        while not done:

            # AI decides which action to take
            action = agent.choose_action(state)

            # Execute action in environment
            next_state, reward, done = env.step(
                action
            )

            # Store experience for future learning
            agent.remember(
                state,
                action,
                reward,
                next_state,
                done
            )

            # Update the current state
            state = next_state

            # Record movement statistics
            episode_steps += 1
            total_steps += 1
            episode_reward += reward

            # ==================================
            # 5. TRAIN THE NEURAL NETWORK
            # ==================================

            # Train every N environment steps
            if total_steps % train_every == 0:

                loss = agent.learn()

                if loss is not None:
                    episode_losses.append(loss)

        # ======================================
        # 6. GAME COMPLETED
        # ======================================

        # Reduce exploration for next episode
        agent.finish_episode()

        score = env.score

        best_score = max(
            best_score,
            score
        )

        # ======================================
        # 7. CALCULATE TRAINING STATISTICS
        # ======================================

        recent_scores = [
            row["score"]
            for row in history[-19:]
        ]

        recent_scores.append(score)

        average_score = (
            sum(recent_scores)
            / len(recent_scores)
        )

        if episode_losses:
            average_loss = (
                sum(episode_losses)
                / len(episode_losses)
            )
        else:
            average_loss = None

        # ======================================
        # 8. STORE EPISODE RESULTS
        # ======================================

        record = {
            "episode": episode,
            "score": score,
            "average_score_20": round(
                average_score, 4
            ),
            "best_score": best_score,
            "steps": episode_steps,
            "epsilon": round(
                agent.epsilon, 6
            ),
            "total_reward": round(
                episode_reward, 4
            ),
            "average_loss": (
                round(average_loss, 6)
                if average_loss is not None
                else ""
            )
        }

        history.append(record)

        # ======================================
        # 9. DISPLAY TRAINING PROGRESS
        # ======================================

        if (
            episode == 1
            or episode % log_every == 0
            or episode == games
        ):

            if average_loss is None:
                loss_text = "N/A"
            else:
                loss_text = f"{average_loss:.4f}"

            print(
                f"Episode {episode:>4}/{games} | "
                f"Score: {score:>2} | "
                f"Avg20: {average_score:>5.2f} | "
                f"Best: {best_score:>2} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Loss: {loss_text}"
            )

    # ======================================
    # 10. SAVE TRAINED MODEL
    # ======================================

    model_path = output_path / "snake_model.npz"

    save_model(
        agent,
        model_path,
        width,
        height,
        games
    )

    # ======================================
    # 11. SAVE TRAINING HISTORY
    # ======================================

    history_path = (
        output_path / "training_history.csv"
    )

    save_history(
        history,
        history_path
    )

    # ======================================
    # 12. FINAL TRAINING SUMMARY
    # ======================================

    final_average = (
        sum(row["score"] for row in history[-20:])
        / min(20, len(history))
    )

    print("\n" + "=" * 55)
    print("TRAINING COMPLETED")
    print("=" * 55)

    print(f"Total episodes: {games}")
    print(f"Total movements: {total_steps}")
    print(f"Best score: {best_score}")
    print(
        f"Final 20-game average: "
        f"{final_average:.2f}"
    )

    print(f"Final epsilon: {agent.epsilon:.4f}")

    print("\nFiles saved:")
    print(f"Model: {model_path}")
    print(f"History: {history_path}")

    print("=" * 55)

    return history, model_path, history_path


# ==========================================
# 13. COMMAND-LINE CONFIGURATION
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description="Train the Snake AI using DQN"
    )

    parser.add_argument(
        "--games",
        type=int,
        default=50
    )

    parser.add_argument(
        "--width",
        type=int,
        default=12
    )

    parser.add_argument(
        "--height",
        type=int,
        default=12
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--train-every",
        type=int,
        default=4
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=10
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts"
    )

    args = parser.parse_args()

    run_training(
        games=args.games,
        width=args.width,
        height=args.height,
        batch_size=args.batch_size,
        train_every=args.train_every,
        log_every=args.log_every,
        seed=args.seed,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()