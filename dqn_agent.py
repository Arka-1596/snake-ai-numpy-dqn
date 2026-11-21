import random
from collections import deque

import numpy as np

from neural_network import NeuralNetwork


# ==========================================
# SNAKE AI - PHASE 4
# DEEP Q-LEARNING AGENT
# ==========================================


class DQNAgent:

    def __init__(
        self,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        learning_rate=0.001,
        memory_size=50000,
        batch_size=16,
        target_sync_interval=200,
        seed=42
    ):

        if not 0 < gamma <= 1:
            raise ValueError("Gamma must be between 0 and 1.")

        if not 0 <= epsilon_min <= epsilon <= 1:
            raise ValueError("Invalid epsilon values.")

        if not 0 < epsilon_decay <= 1:
            raise ValueError("Invalid epsilon decay.")

        if memory_size < batch_size or batch_size < 1:
            raise ValueError("Invalid replay memory settings.")

        if target_sync_interval < 1:
            raise ValueError("Sync interval must be positive.")

        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.target_sync_interval = target_sync_interval

        self.rng = random.Random(seed)

        # Experience replay memory
        self.memory = deque(maxlen=memory_size)

        # Main network: learns from experiences
        self.policy_network = NeuralNetwork(
            input_size=11,
            hidden_size=64,
            output_size=3,
            seed=seed
        )

        # Target network: produces stable targets
        self.target_network = NeuralNetwork(
            input_size=11,
            hidden_size=64,
            output_size=3,
            seed=seed + 1
        )

        self.training_steps = 0

        # Both networks must start with equal weights
        self.sync_target()

    # ======================================
    # 1. SELECT AN ACTION
    # ======================================

    def choose_action(self, state):

        # Exploration: choose a random action
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(3)

        # Exploitation: use the neural network
        q_values = self.policy_network.predict(state)

        return int(np.argmax(q_values))

    # ======================================
    # 2. STORE AN EXPERIENCE
    # ======================================

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        if action not in (0, 1, 2):
            raise ValueError("Invalid action.")

        experience = (
            list(state),
            int(action),
            float(reward),
            list(next_state),
            bool(done)
        )

        self.memory.append(experience)

    # ======================================
    # 3. CALCULATE Q-LEARNING TARGET
    # ======================================

    def compute_target(
        self,
        reward,
        next_state,
        done
    ):

        # No future rewards after a terminal state
        if done:
            return float(reward)

        # Estimate future values using target network
        next_q_values = self.target_network.predict(
            next_state
        )

        max_future_q = float(np.max(next_q_values))

        target_q = (
            float(reward)
            + self.gamma * max_future_q
        )

        return target_q

    # ======================================
    # 4. LEARN FROM REPLAY MEMORY
    # ======================================

    def learn(self):

        # Wait until enough experiences are available
        if len(self.memory) < self.batch_size:
            return None

        # Randomly sample previous experiences
        batch = self.rng.sample(
            self.memory,
            self.batch_size
        )

        losses = []

        for experience in batch:

            (
                state,
                action,
                reward,
                next_state,
                done
            ) = experience

            # Calculate Bellman target
            target_q = self.compute_target(
                reward,
                next_state,
                done
            )

            # Train the main neural network
            loss = self.policy_network.train_step(
                state=state,
                action=action,
                target_q=target_q,
                learning_rate=self.learning_rate
            )

            losses.append(loss)

        self.training_steps += 1

        # Periodically update the target network
        if (
            self.training_steps
            % self.target_sync_interval == 0
        ):
            self.sync_target()

        return float(np.mean(losses))

    # ======================================
    # 5. SYNCHRONIZE TARGET NETWORK
    # ======================================

    def sync_target(self):

        # Copy parameter values, not references
        np.copyto(
            self.target_network.W1,
            self.policy_network.W1
        )

        np.copyto(
            self.target_network.b1,
            self.policy_network.b1
        )

        np.copyto(
            self.target_network.W2,
            self.policy_network.W2
        )

        np.copyto(
            self.target_network.b2,
            self.policy_network.b2
        )

    # ======================================
    # 6. DECAY EXPLORATION RATE
    # ======================================

    def finish_episode(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

        return self.epsilon


# ==========================================
# END OF FILE
# ==========================================