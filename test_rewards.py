
import math

from snake_env import SnakeEnvironment


# ==========================================
# SNAKE AI - REWARD SYSTEM TEST
# ==========================================


def check(name, condition):

    if not condition:
        raise AssertionError(
            f"FAILED: {name}"
        )

    print(f"PASS: {name}")


def main():

    print("\nREWARD OPTIMIZATION TESTS")
    print("=" * 40)

    env = SnakeEnvironment(
        width=12,
        height=12,
        seed=42
    )

    # ======================================
    # TEST 1: APPROACHING FOOD
    # ======================================

    env.reset()

    x, y = env.snake[0]
    env.food = (x + 3, y)

    _, closer_reward, done = env.step(0)

    check(
        "Moving closer receives positive reward",
        math.isclose(
            closer_reward,
            0.14,
            abs_tol=1e-9
        ) and not done
    )

    # ======================================
    # TEST 2: MOVING AWAY
    # ======================================

    env.reset()

    x, y = env.snake[0]
    env.food = (x - 3, y)

    _, farther_reward, done = env.step(0)

    check(
        "Moving away receives negative reward",
        math.isclose(
            farther_reward,
            -0.16,
            abs_tol=1e-9
        ) and not done
    )

    # ======================================
    # TEST 3: REPEATED LOOP
    # ======================================

    env.reset()
    env.food = (0, 0)

    # Four right turns form a square.
    # The head returns to its starting cell.
    for action in (1, 1, 1, 1):

        _, loop_reward, done = env.step(
            action
        )

        if done:
            break

    check(
        "A complete square loop is penalized",
        not done
        and env.snake[0] == (6, 6)
        and math.isclose(
            loop_reward,
            -0.36,
            abs_tol=1e-9
        )
    )

    # ======================================
    # TEST 4: STARVATION
    # ======================================

    env.reset()
    env.food = (0, 0)

    env.moves_without_food = (
        40 * len(env.snake) - 1
    )

    _, reward, done = env.step(0)

    check(
        "Excessive wandering ends the game",
        done
        and reward == -10.0
        and env.termination_reason == "starvation"
    )

    # ======================================
    # TEST 5: FOOD COLLECTION
    # ======================================

    env.reset()

    x, y = env.snake[0]
    env.food = (x + 1, y)

    _, reward, done = env.step(0)

    check(
        "Food still provides +10 reward",
        reward == 10.0
        and env.score == 1
    )

    check(
        "Visit history resets after eating",
        env.visited_since_food == {
            env.snake[0]
        }
    )

    # ======================================
    # TEST 6: STATE COMPATIBILITY
    # ======================================

    state = env.get_state()

    check(
        "Network still receives 11 inputs",
        len(state) == 11
    )

    print("\n" + "=" * 40)
    print("ALL REWARD TESTS PASSED!")
    print("=" * 40)


if __name__ == "__main__":
    main()
