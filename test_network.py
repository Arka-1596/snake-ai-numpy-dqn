import numpy as np

from snake_env import SnakeEnvironment
from neural_network import NeuralNetwork


# ==========================================
# SNAKE AI - PHASE 3 TESTS
# ==========================================


def check(name, condition):

    if not condition:
        raise AssertionError(
            f"FAILED: {name}"
        )

    print(f"PASS: {name}")


def main():

    print("\nSNAKE AI - PHASE 3 TESTS")
    print("=" * 40)

    # ======================================
    # TEST 1: CREATE NETWORK
    # ======================================

    network = NeuralNetwork(
        input_size=11,
        hidden_size=64,
        output_size=3,
        seed=42
    )

    check(
        "Input weights have shape (11, 64)",
        network.W1.shape == (11, 64)
    )

    check(
        "Output weights have shape (64, 3)",
        network.W2.shape == (64, 3)
    )

    # ======================================
    # TEST 2: CONNECT SNAKE ENVIRONMENT
    # ======================================

    env = SnakeEnvironment(seed=42)

    state = env.reset()

    print("\nSnake state:")
    print(state)

    q_values = network.predict(state)

    print("\nInitial Q-values:")
    print(q_values)

    check(
        "Network produces 3 outputs",
        q_values.shape == (3,)
    )

    check(
        "All predicted values are finite",
        np.all(np.isfinite(q_values))
    )

    # ======================================
    # TEST 3: CHECK REPRODUCIBILITY
    # ======================================

    second_network = NeuralNetwork(seed=42)

    second_prediction = second_network.predict(
        state
    )

    check(
        "Same seed gives same initial prediction",
        np.allclose(q_values, second_prediction)
    )

    # ======================================
    # TEST 4: TRAIN ON A FIXED TARGET
    # ======================================

    # This is a temporary training exercise.
    # The target is NOT yet calculated by Q-learning.

    selected_action = 1
    target_q = 5.0

    prediction_before = network.predict(
        state
    )[selected_action]

    initial_error = abs(
        prediction_before - target_q
    )

    losses = []

    for _ in range(200):

        loss = network.train_step(
            state=state,
            action=selected_action,
            target_q=target_q,
            learning_rate=0.001
        )

        losses.append(loss)

    prediction_after = network.predict(
        state
    )[selected_action]

    final_error = abs(
        prediction_after - target_q
    )

    print("\nTRAINING RESULTS")
    print("-" * 40)

    print(
        f"Prediction before: {prediction_before:.4f}"
    )

    print(
        f"Prediction after:  {prediction_after:.4f}"
    )

    print(
        f"Target Q-value:    {target_q:.4f}"
    )

    print(
        f"Initial loss:      {losses[0]:.6f}"
    )

    print(
        f"Final loss:        {losses[-1]:.6f}"
    )

    check(
        "Training reduces prediction error",
        final_error < initial_error
    )

    check(
        "Training reduces loss",
        losses[-1] < losses[0]
    )

    # ======================================
    # TEST 5: PREDICTED ACTION
    # ======================================

    final_q_values = network.predict(state)

    best_action = int(
        np.argmax(final_q_values)
    )

    action_names = [
        "Straight",
        "Right",
        "Left"
    ]

    print("\nFinal Q-values:")
    print(final_q_values)

    print(
        "\nHighest-value action:",
        action_names[best_action]
    )

    print("\n" + "=" * 40)
    print("ALL PHASE 3 TESTS PASSED!")
    print("=" * 40)


if __name__ == "__main__":
    main()