import numpy as np

from snake_env import SnakeEnvironment
from dqn_agent import DQNAgent


# ==========================================
# SNAKE AI - PHASE 4 TESTS
# ==========================================


def check(name, condition):

    if not condition:
        raise AssertionError(f"FAILED: {name}")

    print(f"PASS: {name}")


def networks_match(agent):

    return all(
        np.array_equal(
            getattr(agent.policy_network, name),
            getattr(agent.target_network, name)
        )
        for name in ("W1", "b1", "W2", "b2")
    )


def main():

    print("\nSNAKE AI - PHASE 4 TESTS")
    print("=" * 45)

    # ======================================
    # TEST 1: CREATE THE AI AGENT
    # ======================================

    env = SnakeEnvironment(seed=42)

    agent = DQNAgent(
        seed=42,
        batch_size=4,
        memory_size=10,
        target_sync_interval=2
    )

    state = env.reset()

    check(
        "Agent created successfully",
        agent is not None
    )

    check(
        "Policy and target networks initially match",
        networks_match(agent)
    )

    # ======================================
    # TEST 2: EXPLORATION
    # ======================================

    actions = [
        agent.choose_action(state)
        for _ in range(20)
    ]

    check(
        "Exploration generates valid actions",
        all(action in (0, 1, 2) for action in actions)
    )

    check(
        "Agent begins with full exploration",
        agent.epsilon == 1.0
    )

    # ======================================
    # TEST 3: EXPLOITATION
    # ======================================

    # Create a known prediction for this test
    agent.epsilon = 0.0

    agent.policy_network.W1.fill(0.0)
    agent.policy_network.b1.fill(0.0)

    agent.policy_network.W2.fill(0.0)
    agent.policy_network.b2[:] = [0.0, 2.0, 1.0]

    action = agent.choose_action(state)

    check(
        "Exploitation selects highest Q-value",
        action == 1
    )

    # ======================================
    # TEST 4: BELLMAN TARGET
    # ======================================

    # Assign known Q-values to target network
    agent.target_network.W1.fill(0.0)
    agent.target_network.b1.fill(0.0)

    agent.target_network.W2.fill(0.0)
    agent.target_network.b2[:] = [1.0, 2.0, 3.0]

    expected_target = 1.0 + agent.gamma * 3.0

    actual_target = agent.compute_target(
        reward=1.0,
        next_state=state,
        done=False
    )

    check(
        "Non-terminal Bellman target is correct",
        np.isclose(actual_target, expected_target)
    )

    terminal_target = agent.compute_target(
        reward=-10.0,
        next_state=state,
        done=True
    )

    check(
        "Terminal target ignores future Q-values",
        terminal_target == -10.0
    )

    # ======================================
    # TEST 5: EXPERIENCE REPLAY
    # ======================================

    learner = DQNAgent(
        seed=42,
        batch_size=4,
        memory_size=10,
        target_sync_interval=2
    )

    state = env.reset()

    # Collect real transitions from the game
    for _ in range(8):

        action = learner.choose_action(state)

        next_state, reward, done = env.step(action)

        learner.remember(
            state,
            action,
            reward,
            next_state,
            done
        )

        if done:
            state = env.reset()
        else:
            state = next_state

    check(
        "Replay memory stores experiences",
        len(learner.memory) == 8
    )

    # ======================================
    # TEST 6: LEARNING
    # ======================================

    weights_before = learner.policy_network.W2.copy()
    biases_before = learner.policy_network.b2.copy()

    loss = learner.learn()

    print("\nFirst training loss:", loss)

    check(
        "Training produces a finite loss",
        loss is not None and np.isfinite(loss)
    )

    parameters_changed = (
        not np.array_equal(
            weights_before,
            learner.policy_network.W2
        )
        or not np.array_equal(
            biases_before,
            learner.policy_network.b2
        )
    )

    check(
        "Learning updates network parameters",
        parameters_changed
    )

    # ======================================
    # TEST 7: TARGET SYNCHRONIZATION
    # ======================================

    check(
        "Target network stays fixed before sync",
        not networks_match(learner)
    )

    # Second update triggers synchronization
    learner.learn()

    check(
        "Target network synchronizes correctly",
        networks_match(learner)
    )

    # ======================================
    # TEST 8: EPSILON DECAY
    # ======================================

    old_epsilon = learner.epsilon

    new_epsilon = learner.finish_episode()

    check(
        "Exploration probability decreases",
        new_epsilon < old_epsilon
    )

    check(
        "Epsilon respects minimum value",
        new_epsilon >= learner.epsilon_min
    )

    print("\n" + "=" * 45)
    print("ALL PHASE 4 TESTS PASSED!")
    print("=" * 45)


if __name__ == "__main__":
    main()