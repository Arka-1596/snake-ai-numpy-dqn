
import argparse
from pathlib import Path

import numpy as np
import pygame

from snake_env import SnakeEnvironment
from neural_network import NeuralNetwork


# ==========================================
# SNAKE AI - LIVE NEURAL NETWORK VISUALIZER
# ==========================================

CELL_SIZE = 30
HEADER_HEIGHT = 80
PANEL_WIDTH = 680

BACKGROUND = (11, 17, 23)
BOARD_COLOR = (15, 26, 33)
PANEL_COLOR = (19, 29, 38)
GRID_COLOR = (36, 49, 59)

SNAKE_COLOR = (29, 157, 147)
HEAD_COLOR = (102, 225, 195)
FOOD_COLOR = (250, 82, 110)

TEXT_COLOR = (237, 244, 247)
MUTED_COLOR = (150, 168, 178)
ACCENT_COLOR = (91, 224, 194)

ACTION_NAMES = ["STRAIGHT", "RIGHT", "LEFT"]

STATE_NAMES = [
    "Danger straight",
    "Danger right",
    "Danger left",
    "Moving left",
    "Moving right",
    "Moving up",
    "Moving down",
    "Food left",
    "Food right",
    "Food up",
    "Food down"
]


# ==========================================
# 1. LOAD THE TRAINED MODEL
# ==========================================

def load_model(model_path):

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
                    f"Invalid shape for {name}"
                )

            np.copyto(current, saved)

        width = int(data["board_width"])
        height = int(data["board_height"])
        episodes = int(data["episodes"])

    return network, width, height, episodes


# ==========================================
# 2. VISUALIZATION CLASS
# ==========================================

class SnakeVisualizer:

    def __init__(self, model_path, speed=140, seed=2026):

        self.network, width, height, episodes = (
            load_model(model_path)
        )

        self.env = SnakeEnvironment(
            width=width,
            height=height,
            seed=seed
        )

        self.trained_episodes = episodes

        self.board_width = width * CELL_SIZE
        self.board_height = height * CELL_SIZE

        self.window_width = (
            self.board_width + PANEL_WIDTH
        )

        self.window_height = max(
            720,
            HEADER_HEIGHT + self.board_height + 120
        )

        pygame.init()

        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height)
        )

        pygame.display.set_caption(
            "Snake AI | Live Neural Network"
        )

        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont(
            "consolas", 17
        )

        self.small_font = pygame.font.SysFont(
            "consolas", 14
        )

        self.tiny_font = pygame.font.SysFont(
            "consolas", 12
        )

        self.title_font = pygame.font.SysFont(
            "consolas", 24, bold=True
        )

        self.speed = max(30, min(600, speed))

        self.show_connections = True
        self.best_score = 0

        self.reset()

    # ======================================
    # 3. RESET
    # ======================================

    def reset(self):

        self.state = self.env.reset()

        self.steps = 0
        self.elapsed = 0

        self.paused = False

        self.last_action = None
        self.last_reward = 0.0

    # ======================================
    # 4. EXECUTE AN AI ACTION
    # ======================================

    def ai_step(self):

        if self.env.done:
            return

        q_values = self.network.predict(
            self.state
        )

        action = int(np.argmax(q_values))

        next_state, reward, done = self.env.step(
            action
        )

        self.state = next_state

        self.last_action = action
        self.last_reward = reward

        self.steps += 1

        self.best_score = max(
            self.best_score,
            self.env.score
        )

    # ======================================
    # 5. GET HIDDEN NEURON ACTIVATIONS
    # ======================================

    def get_activations(self):

        x = np.asarray(
            self.state,
            dtype=np.float64
        )

        z1 = x @ self.network.W1 + self.network.b1

        hidden = np.maximum(0, z1)

        outputs = hidden @ self.network.W2
        outputs += self.network.b2

        return hidden, outputs

    # ======================================
    # 6. TEXT HELPER
    # ======================================

    def text(
        self,
        content,
        x,
        y,
        color=TEXT_COLOR,
        font=None
    ):

        if font is None:
            font = self.font

        surface = font.render(
            str(content),
            True,
            color
        )

        self.screen.blit(surface, (int(x), int(y)))

    # ======================================
    # 7. DRAW SNAKE GAME
    # ======================================

    def draw_game(self):

        self.text(
            "SNAKE AI",
            16,
            14,
            ACCENT_COLOR,
            self.title_font
        )

        self.text(
            f"SCORE {self.env.score}   STEPS {self.steps}",
            16,
            52,
            TEXT_COLOR,
            self.small_font
        )

        pygame.draw.rect(
            self.screen,
            BOARD_COLOR,
            (
                0,
                HEADER_HEIGHT,
                self.board_width,
                self.board_height
            )
        )

        # Grid
        for x in range(self.env.width + 1):

            px = x * CELL_SIZE

            pygame.draw.line(
                self.screen,
                GRID_COLOR,
                (px, HEADER_HEIGHT),
                (
                    px,
                    HEADER_HEIGHT + self.board_height
                )
            )

        for y in range(self.env.height + 1):

            py = HEADER_HEIGHT + y * CELL_SIZE

            pygame.draw.line(
                self.screen,
                GRID_COLOR,
                (0, py),
                (self.board_width, py)
            )

        # Food
        if self.env.food is not None:

            fx, fy = self.env.food

            pygame.draw.circle(
                self.screen,
                FOOD_COLOR,
                (
                    fx * CELL_SIZE + CELL_SIZE // 2,
                    HEADER_HEIGHT
                    + fy * CELL_SIZE
                    + CELL_SIZE // 2
                ),
                CELL_SIZE // 3
            )

        # Snake
        for index, (x, y) in enumerate(self.env.snake):

            rect = pygame.Rect(
                x * CELL_SIZE + 2,
                HEADER_HEIGHT + y * CELL_SIZE + 2,
                CELL_SIZE - 4,
                CELL_SIZE - 4
            )

            color = (
                HEAD_COLOR if index == 0
                else SNAKE_COLOR
            )

            pygame.draw.rect(
                self.screen,
                color,
                rect,
                border_radius=5
            )

        if self.env.done:

            overlay = pygame.Surface(
                (
                    self.board_width,
                    self.board_height
                ),
                pygame.SRCALPHA
            )

            overlay.fill((0, 0, 0, 170))

            self.screen.blit(
                overlay,
                (0, HEADER_HEIGHT)
            )

            title = (
                "BOARD FILLED!"
                if self.env.food is None
                else "GAME OVER"
            )

            surface = self.title_font.render(
                title,
                True,
                TEXT_COLOR
            )

            center = (
                self.board_width // 2,
                HEADER_HEIGHT + self.board_height // 2
            )

            self.screen.blit(
                surface,
                surface.get_rect(center=center)
            )

        status_y = (
            HEADER_HEIGHT + self.board_height + 20
        )

        if self.env.done:
            status = "Game ended. Press R to restart."
        elif self.paused:
            status = "PAUSED"
        else:
            status = "Neural network controlling Snake"

        self.text(status, 15, status_y, ACCENT_COLOR)

        self.text(
            f"Best score this session: {self.best_score}",
            15,
            status_y + 34,
            MUTED_COLOR,
            self.small_font
        )

    # ======================================
    # 8. DRAW NETWORK CONNECTIONS
    # ======================================

    def draw_connections(
        self,
        input_positions,
        hidden_positions,
        output_positions
    ):

        if not self.show_connections:
            return

        surface = pygame.Surface(
            (
                self.window_width,
                self.window_height
            ),
            pygame.SRCALPHA
        )

        # Display the two strongest absolute
        # input weights for each hidden neuron.
        for hidden_index, hidden_pos in enumerate(
            hidden_positions
        ):

            weights = self.network.W1[:, hidden_index]

            strongest = np.argsort(
                np.abs(weights)
            )[-2:]

            for input_index in strongest:

                weight = weights[input_index]

                if weight >= 0:
                    color = (54, 184, 164, 42)
                else:
                    color = (205, 105, 155, 42)

                pygame.draw.line(
                    surface,
                    color,
                    input_positions[input_index],
                    hidden_pos,
                    1
                )

        # Display strongest hidden-to-output
        # weights for each output neuron.
        for output_index, output_pos in enumerate(
            output_positions
        ):

            weights = self.network.W2[:, output_index]

            strongest = np.argsort(
                np.abs(weights)
            )[-12:]

            for hidden_index in strongest:

                weight = weights[hidden_index]

                if weight >= 0:
                    color = (54, 184, 164, 65)
                else:
                    color = (205, 105, 155, 65)

                pygame.draw.line(
                    surface,
                    color,
                    hidden_positions[hidden_index],
                    output_pos,
                    1
                )

        self.screen.blit(surface, (0, 0))

    # ======================================
    # 9. DRAW LIVE NEURAL NETWORK
    # ======================================

    def draw_network(self):

        panel_x = self.board_width
        left = panel_x + 20

        pygame.draw.rect(
            self.screen,
            PANEL_COLOR,
            (
                panel_x,
                0,
                PANEL_WIDTH,
                self.window_height
            )
        )

        self.text(
            "LIVE NEURAL NETWORK",
            left,
            18,
            ACCENT_COLOR,
            self.title_font
        )

        self.text(
            "11 INPUTS  >  64 HIDDEN  >  3 OUTPUTS",
            left,
            53,
            MUTED_COLOR,
            self.small_font
        )

        # ==================================
        # DEFINE NODE POSITIONS
        # ==================================

        input_positions = [
            (left + 129, 112 + i * 23)
            for i in range(11)
        ]

        # Eight-by-eight arrangement:
        # all 64 nodes belong to ONE hidden layer.
        hidden_positions = [
            (
                left + 220 + column * 32,
                120 + row * 31
            )
            for row in range(8)
            for column in range(8)
        ]

        output_positions = [
            (left + 520, 160 + i * 75)
            for i in range(3)
        ]

        hidden, q_values = self.get_activations()

        best_action = int(np.argmax(q_values))

        # Connections are drawn behind neurons
        self.draw_connections(
            input_positions,
            hidden_positions,
            output_positions
        )

        # ==================================
        # INPUT NEURONS
        # ==================================

        for index, position in enumerate(
            input_positions
        ):

            value = self.state[index]

            color = (
                ACCENT_COLOR
                if value == 1
                else (66, 80, 91)
            )

            pygame.draw.circle(
                self.screen,
                color,
                position,
                6
            )

            self.text(
                STATE_NAMES[index],
                left + 4,
                position[1] - 8,
                color,
                self.tiny_font
            )

        # ==================================
        # HIDDEN NEURONS
        # ==================================

        maximum = max(
            float(np.max(hidden)),
            1e-8
        )

        for index, position in enumerate(
            hidden_positions
        ):

            activation = float(hidden[index])

            intensity = min(
                1.0,
                activation / maximum
            )

            if activation <= 0:
                color = (43, 55, 65)

            else:
                color = (
                    int(45 + 90 * intensity),
                    int(95 + 130 * intensity),
                    int(105 + 105 * intensity)
                )

            pygame.draw.circle(
                self.screen,
                color,
                position,
                7
            )

        # ==================================
        # OUTPUT NEURONS
        # ==================================

        for index, position in enumerate(
            output_positions
        ):

            selected = index == best_action

            color = (
                ACCENT_COLOR
                if selected
                else (65, 110, 121)
            )

            pygame.draw.circle(
                self.screen,
                color,
                position,
                11
            )

            if selected:
                pygame.draw.circle(
                    self.screen,
                    TEXT_COLOR,
                    position,
                    14,
                    2
                )

            self.text(
                ACTION_NAMES[index],
                position[0] + 18,
                position[1] - 17,
                color,
                self.small_font
            )

            self.text(
                f"Q: {q_values[index]:+.3f}",
                position[0] + 18,
                position[1] + 1,
                MUTED_COLOR,
                self.small_font
            )

        self.text(
            "8 x 8 layout = ONE hidden layer",
            left + 195,
            365,
            MUTED_COLOR,
            self.tiny_font
        )

        self.text(
            "Only strongest-weight connections shown",
            left + 195,
            382,
            MUTED_COLOR,
            self.tiny_font
        )

        # ==================================
        # LOWER INFORMATION PANEL
        # ==================================

        pygame.draw.line(
            self.screen,
            GRID_COLOR,
            (left, 411),
            (self.window_width - 20, 411)
        )

        self.text(
            "CURRENT NETWORK STATE",
            left,
            426,
            ACCENT_COLOR
        )

        active_neurons = int(
            np.count_nonzero(hidden)
        )

        self.text(
            f"Active hidden neurons: {active_neurons}/64",
            left,
            459,
            TEXT_COLOR,
            self.small_font
        )

        self.text(
            f"Trained episodes: {self.trained_episodes}",
            left,
            487,
            TEXT_COLOR,
            self.small_font
        )

        if self.last_action is None:
            last_action = "NONE"
        else:
            last_action = ACTION_NAMES[self.last_action]

        self.text(
            f"Last action: {last_action}",
            left,
            515,
            TEXT_COLOR,
            self.small_font
        )

        self.text(
            f"Last reward: {self.last_reward:+.2f}",
            left,
            543,
            TEXT_COLOR,
            self.small_font
        )

        self.text(
            f"Move delay: {self.speed} ms",
            left,
            571,
            TEXT_COLOR,
            self.small_font
        )

        self.text(
            "Positive weights: teal | Negative: pink",
            left,
            606,
            MUTED_COLOR,
            self.small_font
        )

        pygame.draw.line(
            self.screen,
            GRID_COLOR,
            (left, 639),
            (self.window_width - 20, 639)
        )

        self.text(
            "SPACE Pause   |   R Restart   |   ESC Quit",
            left,
            652,
            MUTED_COLOR,
            self.small_font
        )

        self.text(
            "UP/DOWN Speed   |   N Toggle connections",
            left,
            679,
            MUTED_COLOR,
            self.small_font
        )

    # ======================================
    # 10. RENDER WINDOW
    # ======================================

    def draw(self):

        self.screen.fill(BACKGROUND)

        self.draw_game()
        self.draw_network()

        pygame.display.flip()

    # ======================================
    # 11. MAIN LOOP
    # ======================================

    def run(self):

        running = True

        while running:

            dt = self.clock.tick(60)

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:

                    if event.key == pygame.K_ESCAPE:
                        running = False

                    elif event.key == pygame.K_r:
                        self.reset()

                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused

                    elif event.key == pygame.K_n:
                        self.show_connections = (
                            not self.show_connections
                        )

                    elif event.key == pygame.K_UP:
                        self.speed = max(
                            30,
                            self.speed - 20
                        )

                    elif event.key == pygame.K_DOWN:
                        self.speed = min(
                            600,
                            self.speed + 20
                        )

            if not self.paused and not self.env.done:

                self.elapsed += dt

                if self.elapsed >= self.speed:
                    self.ai_step()
                    self.elapsed = 0

            self.draw()

        pygame.quit()


# ==========================================
# 12. ENTRY POINT
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description="Live Snake AI neural network visualizer"
    )

    parser.add_argument(
        "--model",
        default="artifacts/smoke/snake_model.npz"
    )

    parser.add_argument(
        "--speed",
        type=int,
        default=140
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026
    )

    args = parser.parse_args()

    viewer = SnakeVisualizer(
        model_path=args.model,
        speed=args.speed,
        seed=args.seed
    )

    viewer.run()


if __name__ == "__main__":
    main()
