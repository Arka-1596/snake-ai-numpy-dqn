
from snake_env import (
    SnakeEnvironment,
    RIGHT,
    DOWN
)


def check(name, condition):
    if not condition:
        raise AssertionError(
            f"FAILED: {name}"
        )

    print(f"PASS: {name}")


def main():

    print("\nSNAKE AI - PHASE 2 TESTS")
    print("=" * 35)

    env = SnakeEnvironment(seed=42)

    # ======================================
    # TEST 1: STATE GENERATION
    # ======================================

    state = env.reset()

    print("\nInitial state:")
    print(state)

    check(
        "State contains 11 inputs",
        len(state) == 11
    )

    check(
        "All inputs are binary",
        all(value in (0, 1) for value in state)
    )

    check(
        "Initial direction is right",
        env.direction == RIGHT
    )

    # ======================================
    # TEST 2: FOOD COLLECTION
    # ======================================

    env.reset()

    head_x, head_y = env.snake[0]

    # Place food directly ahead for testing
    env.food = (head_x + 1, head_y)

    initial_length = len(env.snake)

    state, reward, done = env.step(0)

    check(
        "Eating food gives +10 reward",
        reward == 10.0
    )

    check(
        "Eating food increases score",
        env.score == 1
    )

    check(
        "Eating food increases snake length",
        len(env.snake) == initial_length + 1
    )

    # ======================================
    # TEST 3: TURNING RIGHT
    # ======================================

    env.reset()

    state, reward, done = env.step(1)

    check(
        "Right turn changes heading to down",
        env.direction == DOWN
    )

    # ======================================
    # TEST 4: WALL COLLISION
    # ======================================

    env.reset()

    # Keep food away from the test path
    env.food = (0, 0)

    done = False

    for _ in range(env.width + 2):

        state, reward, done = env.step(0)

        if done:
            break

    check(
        "Wall collision ends game",
        done
    )

    check(
        "Wall collision gives -10 reward",
        reward == -10.0
    )

    # ======================================
    # TEST 5: SELF COLLISION
    # ======================================

    env.reset()

    # Controlled snake shape for testing
    env.snake = [
        (5, 5),
        (5, 6),
        (6, 6),
        (6, 5),
        (7, 5)
    ]

    env.direction = RIGHT
    env.food = (0, 0)

    state = env.get_state()

    check(
        "Danger straight is detected",
        state[0] == 1
    )

    state, reward, done = env.step(0)

    check(
        "Self collision ends game",
        done and reward == -10.0
    )

    print("\n" + "=" * 35)
    print("ALL PHASE 2 TESTS PASSED!")
    print("=" * 35)


if __name__ == "__main__":
    main()
