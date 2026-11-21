
import random


# ==========================================
# SNAKE AI - REWARD OPTIMIZATION
# ENVIRONMENT VERSION 2
# ==========================================

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

ACTION_STRAIGHT = 0
ACTION_RIGHT = 1
ACTION_LEFT = 2

# Reward configuration
FOOD_REWARD = 10.0
COLLISION_PENALTY = -10.0

STEP_PENALTY = -0.01
CLOSER_REWARD = 0.15
FARTHER_PENALTY = -0.15
REVISIT_PENALTY = -0.20

# Maximum moves without food:
# This value is multiplied by snake length.
STARVATION_MULTIPLIER = 40


class SnakeEnvironment:

    def __init__(
        self,
        width=20,
        height=20,
        seed=None
    ):

        if width < 5 or height < 5:
            raise ValueError(
                "Grid dimensions must be at least 5."
            )

        self.width = width
        self.height = height
        self.random = random.Random(seed)

        self.reset()

    # ======================================
    # 1. RESET ENVIRONMENT
    # ======================================

    def reset(self):

        center_x = self.width // 2
        center_y = self.height // 2

        self.snake = [
            (center_x, center_y),
            (center_x - 1, center_y),
            (center_x - 2, center_y)
        ]

        self.direction = RIGHT

        self.score = 0
        self.done = False

        self.moves_without_food = 0
        self.termination_reason = None

        # Track positions visited since
        # the last food collection.
        self.visited_since_food = {
            self.snake[0]
        }

        self.food = self._spawn_food()

        return self.get_state()

    # ======================================
    # 2. GENERATE FOOD
    # ======================================

    def _spawn_food(self):

        occupied = set(self.snake)

        available = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in occupied
        ]

        if not available:
            return None

        return self.random.choice(available)

    # ======================================
    # 3. RELATIVE DIRECTION
    # ======================================

    def _turn(self, direction, action):

        dx, dy = direction

        if action == ACTION_STRAIGHT:
            return direction

        if action == ACTION_RIGHT:
            return (-dy, dx)

        if action == ACTION_LEFT:
            return (dy, -dx)

        raise ValueError(
            "Action must be 0, 1, or 2."
        )

    # ======================================
    # 4. COLLISION DETECTION
    # ======================================

    def _is_collision(self, point):

        x, y = point

        if (
            x < 0 or x >= self.width
            or y < 0 or y >= self.height
        ):
            return True

        eating = point == self.food

        # The tail moves away unless
        # the snake is eating food.
        body = (
            self.snake
            if eating
            else self.snake[:-1]
        )

        return point in body

    # ======================================
    # 5. DISTANCE TO FOOD
    # ======================================

    def _food_distance(self, point):

        if self.food is None:
            return 0

        x, y = point
        food_x, food_y = self.food

        # Manhattan distance
        return (
            abs(x - food_x)
            + abs(y - food_y)
        )

    # ======================================
    # 6. GENERATE 11 STATE INPUTS
    # ======================================

    def get_state(self):

        head_x, head_y = self.snake[0]

        straight = self.direction

        right = self._turn(
            self.direction,
            ACTION_RIGHT
        )

        left = self._turn(
            self.direction,
            ACTION_LEFT
        )

        point_straight = (
            head_x + straight[0],
            head_y + straight[1]
        )

        point_right = (
            head_x + right[0],
            head_y + right[1]
        )

        point_left = (
            head_x + left[0],
            head_y + left[1]
        )

        danger_straight = int(
            self._is_collision(point_straight)
        )

        danger_right = int(
            self._is_collision(point_right)
        )

        danger_left = int(
            self._is_collision(point_left)
        )

        direction_left = int(
            self.direction == LEFT
        )

        direction_right = int(
            self.direction == RIGHT
        )

        direction_up = int(
            self.direction == UP
        )

        direction_down = int(
            self.direction == DOWN
        )

        if self.food is not None:

            food_x, food_y = self.food

            food_left = int(
                food_x < head_x
            )

            food_right = int(
                food_x > head_x
            )

            food_up = int(
                food_y < head_y
            )

            food_down = int(
                food_y > head_y
            )

        else:

            food_left = 0
            food_right = 0
            food_up = 0
            food_down = 0

        return [
            danger_straight,
            danger_right,
            danger_left,

            direction_left,
            direction_right,
            direction_up,
            direction_down,

            food_left,
            food_right,
            food_up,
            food_down
        ]

    # ======================================
    # 7. EXECUTE ONE ACTION
    # ======================================

    def step(self, action):

        if self.done:
            raise RuntimeError(
                "Game is over. Call reset() first."
            )

        if action not in (0, 1, 2):
            raise ValueError(
                "Invalid action."
            )

        self.direction = self._turn(
            self.direction,
            action
        )

        head_x, head_y = self.snake[0]
        dx, dy = self.direction

        new_head = (
            head_x + dx,
            head_y + dy
        )

        # ==================================
        # COLLISION
        # ==================================

        if self._is_collision(new_head):

            self.done = True
            self.termination_reason = "collision"

            return (
                self.get_state(),
                COLLISION_PENALTY,
                True
            )

        old_distance = self._food_distance(
            self.snake[0]
        )

        new_distance = self._food_distance(
            new_head
        )

        eating = new_head == self.food

        # Move snake head
        self.snake.insert(0, new_head)

        # ==================================
        # FOOD COLLECTED
        # ==================================

        if eating:

            self.score += 1
            self.moves_without_food = 0

            reward = FOOD_REWARD

            # Start new visit history
            self.visited_since_food = {
                new_head
            }

            self.food = self._spawn_food()

            if self.food is None:

                self.done = True
                self.termination_reason = "board_full"

        # ==================================
        # NORMAL MOVEMENT
        # ==================================

        else:

            self.snake.pop()

            reward = STEP_PENALTY

            self.moves_without_food += 1

            # Reward movement toward food
            if new_distance < old_distance:

                reward += CLOSER_REWARD

            # Penalize movement away from food
            elif new_distance > old_distance:

                reward += FARTHER_PENALTY

            # Discourage repeatedly visiting
            # the same cells without food.
            if new_head in self.visited_since_food:

                reward += REVISIT_PENALTY

            self.visited_since_food.add(
                new_head
            )

            # ==================================
            # STARVATION CHECK
            # ==================================

            move_limit = (
                STARVATION_MULTIPLIER
                * len(self.snake)
            )

            if self.moves_without_food >= move_limit:

                self.done = True

                self.termination_reason = (
                    "starvation"
                )

                reward = COLLISION_PENALTY

        next_state = self.get_state()

        return (
            next_state,
            reward,
            self.done
        )


# ==========================================
# END OF FILE
# ==========================================
