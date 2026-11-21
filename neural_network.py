import numpy as np


# ==========================================
# SNAKE AI - PHASE 3
# NEURAL NETWORK BUILT USING NUMPY
# ==========================================


class NeuralNetwork:

    def __init__(
        self,
        input_size=11,
        hidden_size=64,
        output_size=3,
        seed=42
    ):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        rng = np.random.default_rng(seed)

        # ==================================
        # 1. INITIALIZE WEIGHTS AND BIASES
        # ==================================

        # Input layer -> Hidden layer
        self.W1 = rng.standard_normal(
            (input_size, hidden_size)
        ) * np.sqrt(2.0 / input_size)

        self.b1 = np.zeros(hidden_size)

        # Hidden layer -> Output layer
        self.W2 = rng.standard_normal(
            (hidden_size, output_size)
        ) * np.sqrt(2.0 / hidden_size)

        self.b2 = np.zeros(output_size)

    # ======================================
    # 2. RELU ACTIVATION FUNCTION
    # ======================================

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    # ======================================
    # 3. VALIDATE INPUT STATE
    # ======================================

    def prepare_state(self, state):

        state = np.asarray(
            state,
            dtype=np.float64
        )

        if state.shape != (self.input_size,):
            raise ValueError(
                f"Expected {self.input_size} inputs, "
                f"received shape {state.shape}"
            )

        if not np.all(np.isfinite(state)):
            raise ValueError(
                "State contains invalid numerical values."
            )

        return state

    # ======================================
    # 4. FORWARD PROPAGATION
    # ======================================

    def forward(self, state):

        x = self.prepare_state(state)

        # Input -> Hidden
        z1 = x @ self.W1 + self.b1

        # Apply ReLU
        a1 = self.relu(z1)

        # Hidden -> Output
        q_values = a1 @ self.W2 + self.b2

        return q_values

    # ======================================
    # 5. PREDICT Q-VALUES
    # ======================================

    def predict(self, state):

        return self.forward(state)

    # ======================================
    # 6. BACKPROPAGATION
    # ======================================

    def train_step(
        self,
        state,
        action,
        target_q,
        learning_rate=0.001
    ):

        if not isinstance(action, (int, np.integer)):
            raise ValueError("Action must be an integer.")

        if not 0 <= action < self.output_size:
            raise ValueError("Invalid action index.")

        if not np.isfinite(target_q):
            raise ValueError("Target Q-value must be finite.")

        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")

        x = self.prepare_state(state)

        # ==================================
        # STEP A: FORWARD PASS
        # ==================================

        z1 = x @ self.W1 + self.b1
        a1 = self.relu(z1)

        q_values = a1 @ self.W2 + self.b2

        predicted_q = q_values[action]

        # ==================================
        # STEP B: CALCULATE LOSS
        # ==================================

        error = predicted_q - target_q

        loss = error ** 2

        # ==================================
        # STEP C: OUTPUT LAYER GRADIENT
        # ==================================

        d_output = np.zeros(self.output_size)

        # Only the chosen action has a target
        d_output[action] = 2 * error

        dW2 = np.outer(a1, d_output)
        db2 = d_output

        # ==================================
        # STEP D: HIDDEN LAYER GRADIENT
        # ==================================

        # Use the original W2 before updating it
        d_hidden = self.W2 @ d_output

        # Derivative of ReLU
        d_relu = (z1 > 0).astype(np.float64)

        d_z1 = d_hidden * d_relu

        dW1 = np.outer(x, d_z1)
        db1 = d_z1

        # ==================================
        # STEP E: UPDATE WEIGHTS
        # ==================================

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

        return float(loss)


# ==========================================
# END OF FILE
# ==========================================