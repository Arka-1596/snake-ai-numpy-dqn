
import argparse
from pathlib import Path

import numpy as np
import pygame

from snake_env import SnakeEnvironment
from neural_network import NeuralNetwork


# ==========================================
# SNAKE AI - PHASE 6
# WATCH THE TRAINED NEURAL NETWORK PLAY
# ==========================================

CELL_SIZE = 30
HUD_HEIGHT = 80
SIDE_PANEL_WIDTH = 300

BACKGROUND = (12, 17, 23)
BOARD_COLOR = (14, 25, 32)
GRID_COLOR = (31, 47, 56)
PANEL_COLOR = (21, 32, 41)

SNAKE_COLOR = (29, 170, 155)
HEAD_COLOR = (105, 230, 195)
FOOD_COLOR = (255, 80, 110)

TEXT_COLOR = (240, 245, 250)
MUTED_COLOR = (158, 178, 188)
ACCENT_COLOR = (105, 230, 195)

ACTION_NAMES = [
    "STRAIGHT",
    "RIGHT",
    "LEFT"
]

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
# 1. LOAD TRAINED NEURAL NETWORK
# ==========================================

def load_model(model_path):

    if not Path(model_path).is_file():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
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

class SnakeAIViewer:

    def __init__(
        self,
        model_path,
        speed=140,
        seed=2026
    ):

        self.network, width, height, trained_games = (
            load_model(model_path)
        )

        pygame.init()

        self.env = SnakeEnvironment(
            width=width,
            height=height,
            seed=seed
        )

        self.board_width = width * CELL_SIZE
        self.board_height = height * CELL_SIZE

        self.window_width = (
            self.board_width + SIDE_PANEL_WIDTH
        )

        self.window_height = max(
            HUD_HEIGHT + self.board_height,
            620
        )

        self.screen = pygame.display.set_mode(
            (
                self.window_width,
                self.window_height
            )
        )

        pygame.display.set_caption(
            "Snake AI - Neural Network Player"
        )

        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont(
            "consolas", 18
        )

        self.small_font = pygame.font.SysFont(
            "consolas", 15
        )

        self.title_font = pygame.font.SysFont(
            "consolas", 24, bold=True
        )

        self.trained_games = trained_games

        self.speed = max(30, min(600, speed))
        self.elapsed = 0

        self.paused = False

        self.reset_game()

    # ======================================
    # 3. RESET GAME
    # ======================================

    def reset_game(self):

        self.state = self.env.reset()

        self.steps = 0
        self.last_action = None
        self.last_reward = 0.0

        self.elapsed = 0
        self.paused = False

    # ======================================
    # 4. EXECUTE AI DECISION
    # ======================================

    def ai_step(self):

        if self.env.done:
            return

        # Neural network predicts three Q-values
        q_values = self.network.predict(
            self.state
        )

        # Greedy action selection
        action = int(np.argmax(q_values))

        # Execute action in the environment
        next_state, reward, done = self.env.step(
            action
        )

        self.state = next_state
        self.last_action = action
        self.last_reward = reward

        self.steps += 1

    # ======================================
    # 5. DRAW TEXT
    # ======================================

    def draw_text(
        self,
        text,
        x,
        y,
        color=TEXT_COLOR,
        font=None
    ):

        if font is None:
            font = self.font

        surface = font.render(
            str(text),
            True,
            color
        )

        self.screen.blit(
            surface,
            (x, y)
        )

    # ======================================
    # 6. DRAW GAME BOARD
    # ======================================

    def draw_board(self):

        board_rect = pygame.Rect(
            0,
            HUD_HEIGHT,
            self.board_width,
            self.board_height
        )

        pygame.draw.rect(
            self.screen,
            BOARD_COLOR,
            board_rect
        )

        # Grid
        for x in range(self.env.width + 1):

            pixel_x = x * CELL_SIZE

            pygame.draw.line(
                self.screen,
                GRID_COLOR,
                (pixel_x, HUD_HEIGHT),
                (
                    pixel_x,
                    HUD_HEIGHT + self.board_height
                )
            )

        for y in range(self.env.height + 1):

            pixel_y = HUD_HEIGHT + y * CELL_SIZE

            pygame.draw.line(
                self.screen,
                GRID_COLOR,
                (0, pixel_y),
                (self.board_width, pixel_y)
            )

        # Food
        if self.env.food is not None:

            food_x, food_y = self.env.food

            center = (
                food_x * CELL_SIZE + CELL_SIZE // 2,
                HUD_HEIGHT + food_y * CELL_SIZE
                + CELL_SIZE // 2
            )

            pygame.draw.circle(
                self.screen,
                FOOD_COLOR,
                center,
                CELL_SIZE // 3
            )

        # Snake
        for index, (x, y) in enumerate(self.env.snake):

            rectangle = pygame.Rect(
                x * CELL_SIZE + 2,
                HUD_HEIGHT + y * CELL_SIZE + 2,
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
                rectangle,
                border_radius=5
            )

        # Game over overlay
        if self.env.done:

            overlay = pygame.Surface(
                (
                    self.board_width,
                    self.board_height
                ),
                pygame.SRCALPHA
            )

            overlay.fill((0, 0, 0, 175))

            self.screen.blit(
                overlay,
                (0, HUD_HEIGHT)
            )

            message = (
                "BOARD FILLED!"
                if self.env.food is None
                else "GAME OVER"
            )

            surface = self.title_font.render(
                message,
                True,
                TEXT_COLOR
            )

            center = (
                self.board_width // 2,
                HUD_HEIGHT + self.board_height // 2
            )

            self.screen.blit(
                surface,
                surface.get_rect(center=center)
            )

    # ======================================
    # 7. DRAW INFORMATION PANEL
    # ======================================

    def draw_panel(self):

        panel_x = self.board_width

        pygame.draw.rect(
            self.screen,
            PANEL_COLOR,
            (
                panel_x,
                0,
                SIDE_PANEL_WIDTH,
                self.window_height
            )
        )

        x = panel_x + 16

        self.draw_text(
            "NEURAL NETWORK",
            x,
            22,
            ACCENT_COLOR,
            self.font
        )

        self.draw_text(
            f"Trained: {self.trained_games} games",
            x,
            52,
            MUTED_COLOR,
            self.small_font
        )

        self.draw_text(
            "MODE: GREEDY",
            x,
            97,
            ACCENT_COLOR,
            self.small_font
        )

        self.draw_text(
            "Epsilon: 0.00",
            x,
            119,
            MUTED_COLOR,
            self.small_font
        )

        # Predict for current state
        q_values = self.network.predict(
            self.state
        )

        best_action = int(
            np.argmax(q_values)
        )

        self.draw_text(
            "CURRENT Q-VALUES",
            x,
            160,
            TEXT_COLOR,
            self.font
        )

        for index, action_name in enumerate(
            ACTION_NAMES
        ):

            y = 190 + index * 24

            color = (
                ACCENT_COLOR
                if index == best_action
                else MUTED_COLOR
            )

            self.draw_text(
                f"{action_name:<9} {q_values[index]:>7.3f}",
                x,
                y,
                color,
                self.small_font
            )

        self.draw_text(
            "11 STATE INPUTS",
            x,
            282,
            TEXT_COLOR,
            self.font
        )

        for index, feature_name in enumerate(
            STATE_NAMES
        ):

            y = 310 + index * 21

            value = self.state[index]

            color = (
                ACCENT_COLOR
                if value == 1
                else MUTED_COLOR
            )

            self.draw_text(
                f"{feature_name:<18} {value}",
                x,
                y,
                color,
                self.small_font
            )

        self.draw_text(
            "SPACE Pause  |  R Restart",
            x,
            557,
            MUTED_COLOR,
            self.small_font
        )

        self.draw_text(
            "UP/DOWN Speed | ESC Quit",
            x,
            581,
            MUTED_COLOR,
            self.small_font
        )

    # ======================================
    # 8. DRAW HEADER AND STATUS
    # ======================================

    def draw_header(self):

        self.draw_text(
            "SNAKE AI",
            15,
            12,
            ACCENT_COLOR,
            self.title_font
        )

        self.draw_text(
            f"SCORE: {self.env.score}   "
            f"STEPS: {self.steps}",
            15,
            50,
            TEXT_COLOR,
            self.small_font
        )

        status_y = (
            HUD_HEIGHT + self.board_height + 16
        )

        if self.env.done:
            status = "Game ended. Press R to restart."
        elif self.paused:
            status = "PAUSED"
        else:
            status = "AI controlling the snake..."

        self.draw_text(
            status,
            15,
            status_y,
            TEXT_COLOR,
            self.small_font
        )

        if self.last_action is not None:

            self.draw_text(
                "Last action: "
                + ACTION_NAMES[self.last_action],
                15,
                status_y + 28,
                MUTED_COLOR,
                self.small_font
            )

            self.draw_text(
                f"Last reward: {self.last_reward:.2f}",
                15,
                status_y + 53,
                MUTED_COLOR,
                self.small_font
            )

        self.draw_text(
            f"Move delay: {self.speed} ms",
            15,
            self.window_height - 25,
            MUTED_COLOR,
            self.small_font
        )

    # ======================================
    # 9. RENDER THE WINDOW
    # ======================================

    def draw(self):

        self.screen.fill(BACKGROUND)

        self.draw_board()
        self.draw_header()
        self.draw_panel()

        pygame.display.flip()

    # ======================================
    # 10. MAIN VISUALIZATION LOOP
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
                        self.reset_game()

                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused

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
# 11. COMMAND-LINE ENTRY POINT
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description="Watch a trained Snake AI"
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

    viewer = SnakeAIViewer(
        model_path=args.model,
        speed=args.speed,
        seed=args.seed
    )

    viewer.run()


if __name__ == "__main__":
    main()
