
import pygame
import random

# ==========================================
# 1. GAME CONFIGURATION
# ==========================================

CELL_SIZE = 25
GRID_WIDTH = 20
GRID_HEIGHT = 20

SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE
HUD_HEIGHT = 60
MOVE_DELAY = 120  # Milliseconds between snake movements

BACKGROUND = (12, 17, 23)
GRID_COLOR = (30, 40, 48)
SNAKE_COLOR = (29, 170, 155)
HEAD_COLOR = (105, 230, 195)
FOOD_COLOR = (255, 80, 110)
TEXT_COLOR = (240, 245, 250)

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)


class SnakeGame:
    def __init__(self):
        pygame.init()

        self.screen = pygame.display.set_mode(
            (SCREEN_WIDTH, SCREEN_HEIGHT + HUD_HEIGHT)
        )
        pygame.display.set_caption("Snake AI - Phase 1")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 22, bold=True)
        self.small_font = pygame.font.SysFont("consolas", 15)

        self.reset()

    def reset(self):
        """Start a new game with a three-segment snake."""
        center_x = GRID_WIDTH // 2
        center_y = GRID_HEIGHT // 2

        self.snake = [
            (center_x, center_y),
            (center_x - 1, center_y),
            (center_x - 2, center_y),
        ]

        self.direction = RIGHT
        self.next_direction = RIGHT
        self.turn_used = False

        self.score = 0
        self.game_over = False
        self.won = False
        self.elapsed = 0
        self.food = self.spawn_food()

    def spawn_food(self):
        """Choose a free grid cell for the next food item."""
        empty_cells = [
            (x, y)
            for x in range(GRID_WIDTH)
            for y in range(GRID_HEIGHT)
            if (x, y) not in self.snake
        ]

        if not empty_cells:
            return None

        return random.choice(empty_cells)

    def change_direction(self, new_direction):
        """Accept a turn while preventing instant reversal."""
        if self.game_over or self.turn_used:
            return

        opposite = (-self.direction[0], -self.direction[1])
        if new_direction == opposite or new_direction == self.direction:
            return

        self.next_direction = new_direction
        self.turn_used = True

    def step(self):
        """Move the snake one cell and check for food or collisions."""
        if self.game_over:
            return

        self.direction = self.next_direction
        self.turn_used = False

        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)
        x, y = new_head

        # Wall collision
        if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT:
            self.game_over = True
            return

        eating = new_head == self.food

        # Moving into the old tail position is allowed if the tail moves away.
        body_to_check = self.snake if eating else self.snake[:-1]
        if new_head in body_to_check:
            self.game_over = True
            return

        self.snake.insert(0, new_head)

        if eating:
            self.score += 1
            self.food = self.spawn_food()

            if self.food is None:
                self.won = True
                self.game_over = True
        else:
            self.snake.pop()

    def draw(self):
        """Draw the score, board, snake, food, and game-over message."""
        self.screen.fill(BACKGROUND)

        score_text = self.font.render(
            f"SCORE: {self.score}", True, TEXT_COLOR
        )
        self.screen.blit(score_text, (15, 17))

        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                rect = pygame.Rect(
                    x * CELL_SIZE,
                    y * CELL_SIZE + HUD_HEIGHT,
                    CELL_SIZE,
                    CELL_SIZE,
                )
                pygame.draw.rect(self.screen, GRID_COLOR, rect, 1)

        if self.food is not None:
            food_x, food_y = self.food
            food_center = (
                food_x * CELL_SIZE + CELL_SIZE // 2,
                food_y * CELL_SIZE + HUD_HEIGHT + CELL_SIZE // 2,
            )
            pygame.draw.circle(
                self.screen, FOOD_COLOR, food_center, CELL_SIZE // 3
            )

        for index, (x, y) in enumerate(self.snake):
            rect = pygame.Rect(
                x * CELL_SIZE + 2,
                y * CELL_SIZE + HUD_HEIGHT + 2,
                CELL_SIZE - 4,
                CELL_SIZE - 4,
            )
            color = HEAD_COLOR if index == 0 else SNAKE_COLOR
            pygame.draw.rect(self.screen, color, rect, border_radius=5)

        if self.game_over:
            overlay = pygame.Surface(
                (SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA
            )
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, HUD_HEIGHT))

            title = "YOU WIN!" if self.won else "GAME OVER"
            message = self.font.render(title, True, TEXT_COLOR)
            restart = self.small_font.render(
                "Press R to restart", True, TEXT_COLOR
            )
            center_y = HUD_HEIGHT + SCREEN_HEIGHT // 2

            self.screen.blit(
                message,
                message.get_rect(
                    center=(SCREEN_WIDTH // 2, center_y - 20)
                ),
            )
            self.screen.blit(
                restart,
                restart.get_rect(
                    center=(SCREEN_WIDTH // 2, center_y + 20)
                ),
            )

        pygame.display.flip()

    def run(self):
        """Read keyboard input, move the snake, and refresh the window."""
        running = True

        key_to_direction = {
            pygame.K_UP: UP,
            pygame.K_DOWN: DOWN,
            pygame.K_LEFT: LEFT,
            pygame.K_RIGHT: RIGHT,
            pygame.K_w: UP,
            pygame.K_s: DOWN,
            pygame.K_a: LEFT,
            pygame.K_d: RIGHT,
        }

        while running:
            dt = self.clock.tick(60)
            self.elapsed += dt

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self.reset()
                    elif event.key in key_to_direction:
                        self.change_direction(
                            key_to_direction[event.key]
                        )

            if not self.game_over and self.elapsed >= MOVE_DELAY:
                self.step()
                self.elapsed = 0

            self.draw()

        pygame.quit()


if __name__ == "__main__":
    game = SnakeGame()
    game.run()
