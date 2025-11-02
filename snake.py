import pygame
import random
import sys
import time
import nn
import matplotlib.pyplot as plt
scores = []

DIMENSIONS = [30, 30]
grid = [[0 for i in range(DIMENSIONS[1])] for j in range(DIMENSIONS[0])]
snake = [[5, 5], [5, 4]]
direction_changes = []

headpointer = [5, 5]
tailpointer = [5, 3]
apple = [5, 7]
current_direction = "d"
tail_direction = "d"

# pygame
pygame.init()
CELL_SIZE = 20
WIDTH, HEIGHT = DIMENSIONS[0] * CELL_SIZE, DIMENSIONS[1] * CELL_SIZE
screen = pygame.display.set_mode((WIDTH + 200, HEIGHT))
pygame.display.set_caption("Snake")
clock = pygame.time.Clock()

colors = {
    0: (0, 15, 30),    # empty
    1: (0, 180, 0),     # body
    2: (0, 255, 0),     # head
    3: (255, 50, 50),   # apple
}

def draw_grid():
    screen.fill((0, 20, 40))
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            color = colors.get(grid[i][j], (255, 255, 255))
            pygame.draw.rect(screen, color, (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE - 2, CELL_SIZE - 2))
    pygame.display.flip()

def draw_input_layer(surface, inputs, x, y, size):
    # draw vertical column representing NN layer
    if len(inputs) != 4: # hard-coded, if the inputs list is not the outputs
        if len(inputs) == 24:
            print (inputs)
        for i, val in enumerate(inputs):
            brightness = max(0, min(255, int(val * 255)))
            color = (int(brightness*0.1), int(brightness*0.6), int(brightness*0.9))
            rect = pygame.Rect(x, y + i * size, size, size)
            pygame.draw.rect(surface, color, rect)
    else: 
        for i, val in enumerate(inputs):
            brightness = max(0, min(255, int(val * 255)))
            color = (int(brightness*0.1), int(brightness*0.6), int(brightness*0.9))
            rect = pygame.Rect(x, y + i * size, size, size)
            pygame.draw.rect(surface, color, rect)

def playRound():
    global headpointer, tailpointer, apple, current_direction, tail_direction
    score = 0
    bestmove = 3
    prev_state = nn.generateInputLayer(grid, headpointer, apple)
    prev_action = 0
    

    while True:
        # make sure the apple cell is marked on the grid so the eating branch detects it
        grid[apple[0]][apple[1]] = 3
        pygame.event.pump()
        for event in pygame.event.get():
            
            if event.type == pygame.QUIT:
                return score

                break
        else:
            # no key pressed, skip loop
            x = current_direction

        if bestmove == 0 and x != "s":
            x = "w"
        elif bestmove == 1 and x != "w":
            x = "s"
        elif bestmove == 2 and x != "d":
            x = "a"
        elif bestmove == 3 and x != "a":
            x = "d"
        else:
            x = current_direction

        grid[apple[0]][apple[1]] = 3
        assert x in "wasd" and len(x) == 1

        if x != current_direction:
            direction_changes.append([headpointer.copy(), x])

        current_direction = x

        # collision check

        if (
            (x == "d" and headpointer[1] + 1 == DIMENSIONS[0])
            or (x == "a" and headpointer[1] == 0)
            or (x == "w" and headpointer[0] == 0)
            or (x == "s" and headpointer[0] + 1 == DIMENSIONS[1])
        ):
            nn.train_Q_network(prev_state, inputs, prev_action, -1.0, True)
            return score
        elif(
            (x == "d" and grid[headpointer[0]][headpointer[1] + 1] == 1)
            or (x == "a" and grid[headpointer[0]][headpointer[1] - 1] == 1)
            or (x == "w" and grid[headpointer[0] - 1][headpointer[1]] == 1)
            or (x == "s" and grid[headpointer[0] + 1][headpointer[1]] == 1)
        ):
            nn.train_Q_network(prev_state, inputs, prev_action, -1.0, True)
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

        ate = False
        score += 0.1

        if grid[headpointer[0]][headpointer[1]] == 3:
            ate = True                             
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
        
        _, h1, _, h2, _, out = nn.feed_forward(inputs)

        # draw the 3 columns side-by-side on the right
        draw_input_layer(screen, inputs, WIDTH + 20, 20, 6)
        draw_input_layer(screen, h1, WIDTH + 40, 20, 6)
        draw_input_layer(screen, h2, WIDTH + 60, 20, 6)
        draw_input_layer(screen, out, WIDTH + 80, 20, 6)

        bestmove = nn.choose_action(out, epsilon=0.1)

        reward = -0.01
        done = False

        if ate:            
            reward = 1.0
        elif headpointer[0] < 0 or headpointer[0] >= DIMENSIONS[0] or headpointer[1] < 0 or headpointer[1] >= DIMENSIONS[1]:
            reward = -1.0
            done = True

        nn.train_Q_network(prev_state, inputs, prev_action, reward, done)

        prev_state = inputs
        prev_action = bestmove
        #print(bestmove)
        pygame.display.flip()
        pygame.display.flip()
        clock.tick(15)

def reset_game():
    global grid, snake, direction_changes, headpointer, tailpointer, apple, current_direction, tail_direction
    grid = [[0 for i in range(DIMENSIONS[1])] for j in range(DIMENSIONS[0])]
    snake = [[5, 5], [5, 4]]
    direction_changes = []
    headpointer = [5, 5]
    tailpointer = [5, 3]
    apple = [5, 7]
    current_direction = "d"
    tail_direction = "d"

episodes = 100  # number of games to train
for episode in range(episodes):
    reset_game()
    start = time.time()
    score = playRound()
    now = time.time()

    eval = score
    print(f"Episode {episode+1}/{episodes} | Score: {score} | Eval: {eval:.2f}")
    scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(scores, label="Score per Episode", linewidth=1)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("Snake RL Training Progress")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

pygame.quit()
sys.exit()