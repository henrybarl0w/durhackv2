import pygame
import random
import sys
import time
import nn

# === Your grid setup ===
DIMENSIONS = [30, 30]
grid = [[0 for i in range(DIMENSIONS[1])] for j in range(DIMENSIONS[0])]
snake = [[5, 5], [5, 4]]
direction_changes = []

headpointer = [5, 5]
tailpointer = [5, 3]
apple = [5, 7]
current_direction = "d"
tail_direction = "d"

# === Pygame setup ===
pygame.init()
CELL_SIZE = 20
WIDTH, HEIGHT = DIMENSIONS[0] * CELL_SIZE, DIMENSIONS[1] * CELL_SIZE
screen = pygame.display.set_mode((WIDTH + 200, HEIGHT))
pygame.display.set_caption("Snake - Your Logic")
clock = pygame.time.Clock()

colors = {
    0: (0, 15, 30),    # empty
    1: (0, 180, 0),     # body
    2: (0, 255, 0),     # head
    3: (255, 50, 50),   # apple
}

# === Functions ===
def draw_grid():
    screen.fill((0, 20, 40))
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            color = colors.get(grid[i][j], (255, 255, 255))
            pygame.draw.rect(screen, color, (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE - 2, CELL_SIZE - 2))
    pygame.display.flip()

def draw_input_layer(surface, inputs, x, y, size):
    """
    Draws a vertical column of squares representing input activations.
    surface: pygame.Surface to draw on
    inputs: list of floats (0-1)
    x, y: top-left corner
    size: side length of each square
    """
    for i, val in enumerate(inputs):
        brightness = max(0, min(255, int(val * 255)))
        color = (int(brightness*0.1), int(brightness*0.6), int(brightness*0.9))
        rect = pygame.Rect(x, y + i * size, size, size)
        pygame.draw.rect(surface, color, rect)


# === Game loop ===
def playRound():
    global headpointer, tailpointer, apple, current_direction, tail_direction
    score = 0
    while True:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return score

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    x = "w"
                elif event.key == pygame.K_s:
                    x = "s"
                elif event.key == pygame.K_a:
                    x = "a"
                elif event.key == pygame.K_d:
                    x = "d"
                elif event.key == pygame.K_q:
                    return score
                else:
                    x = current_direction
                break
        else:
            # no key pressed, skip loop
            x = current_direction

        grid[apple[0]][apple[1]] = 3
        assert x in "wasd" and len(x) == 1

        if x != current_direction:
            direction_changes.append([headpointer.copy(), x])

        current_direction = x

        # collision check
        try:
            if x == "d" and grid[headpointer[0]][headpointer[1] + 1] == 1:
                return score
            if x == "a" and grid[headpointer[0]][headpointer[1] - 1] == 1:
                return score
            if x == "w" and grid[headpointer[0] - 1][headpointer[1]] == 1:
                return score
            if x == "s" and grid[headpointer[0] + 1][headpointer[1]] == 1:
                return score
        except:
            return score

        grid[headpointer[0]][headpointer[1]] = 1

        if x == "d":
            headpointer[1] += 1
        elif x == "a":
            headpointer[1] -= 1
        elif x == "w":
            headpointer[0] -= 1
        elif x == "s":
            headpointer[0] += 1

        snake.append(headpointer.copy())

        if grid[headpointer[0]][headpointer[1]] == 3:
            apple = random.choice([[i, j] for i in range(DIMENSIONS[0]) for j in range(DIMENSIONS[1]) if [i, j] not in snake])
            grid[headpointer[0]][headpointer[1]] = 2
            score += 1
        else:
            grid[headpointer[0]][headpointer[1]] = 2

            if len(direction_changes) != 0:
                if direction_changes[0][0] == tailpointer:
                    tail_direction = direction_changes[0][1]
                    direction_changes.pop(0)

            x = tail_direction
            if x == "d":
                tailpointer[1] += 1
            elif x == "a":
                tailpointer[1] -= 1
            elif x == "w":
                tailpointer[0] -= 1
            elif x == "s":
                tailpointer[0] += 1

            grid[tailpointer[0]][tailpointer[1]] = 0

        draw_grid()
        inputs = nn.generateInputLayer(grid, headpointer, apple)
        draw_input_layer(screen, inputs, WIDTH + 20, 20, 6)
        inputs = nn.generateInputLayer(grid, headpointer, apple)
        h1, h2, out = nn.feed_forward(inputs)

        # draw the 3 columns side-by-side on the right
        draw_input_layer(screen, inputs, WIDTH + 20, 20, 6)
        draw_input_layer(screen, h1, WIDTH + 40, 20, 6)
        draw_input_layer(screen, h2, WIDTH + 60, 20, 6)
        draw_input_layer(screen, out, WIDTH + 80, 20, 6)

        pygame.display.flip()
        pygame.display.flip()
        clock.tick(15)
start = time.time()
x = playRound()
now = time.time()

eval = x ** 2 / (now - start)
print(x)
pygame.quit()
sys.exit()