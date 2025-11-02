import random
import math

def he_init(n_inputs, n_outputs):
    std = math.sqrt(2 / n_inputs)
    return [[random.gauss(0, std) for _ in range(n_outputs)] for _ in range(n_inputs)]


def generateInputLayer(board, head, apple):
    # board + head + apple
    input_layer = []

    rows = len(board)
    cols = len(board[0])
    hr, hc = head
    ar, ac = apple

    # --- 7x7 grid around head ---
    for dr in range(-3, 4):
        for dc in range(-3, 4):
            r, c = hr + dr, hc + dc
            if 0 <= r < rows and 0 <= c < cols:
                input_layer.append(board[r][c])
            else:
                # outside bounds = treat as wall
                input_layer.append(1)

    # --- distances to walls (normalized 0–1) ---
    dist_top = hr / rows
    dist_bottom = (rows - hr - 1) / rows
    dist_left = hc / cols
    dist_right = (cols - hc - 1) / cols
    input_layer.extend([dist_top, dist_bottom, dist_left, dist_right])

    # --- relative position of apple (normalized) ---
    apple_vert = (ar - hr) / rows    # positive = apple below
    apple_horiz = (ac - hc) / cols   # positive = apple to right
    input_layer.extend([apple_vert, apple_horiz])

    #print(len(input_layer))

    return input_layer

# ARCHITECTURE: 55 x 24 x 16 x 4

# weights: 3D array. weights[layer][neuron][connectsto]

random.seed(57)

weights = [
    he_init(55, 24),
    he_init(24, 16),
    he_init(16, 4)
]


# biases: 2D array: biases[layer][neuron]
biases = [
    [], 
    [random.random() for j in range(24)],
    [random.random() for j in range(16)],
    [random.random() for j in range(4)]
]

def feed_forward(inputLayer):
    """Return (z1, a1, z2, a2, z3, a3) where a3 are raw Q values (linear)."""
    # Layer 1 (input -> 24)
    z1 = [0.0] * 24
    a1 = [0.0] * 24
    for j in range(24):
        total = biases[1][j]
        for i in range(len(inputLayer)):
            total += inputLayer[i] * weights[0][i][j]
        z1[j] = total
        a1[j] = relu(total)

    # Layer 2 (24 -> 16)
    z2 = [0.0] * 16
    a2 = [0.0] * 16
    for j in range(16):
        total = biases[2][j]
        for i in range(24):
            total += a1[i] * weights[1][i][j]   # <- IMPORTANT: use a1 (previous activation)
        z2[j] = total
        a2[j] = relu(total)

    # Output layer (16 -> 4), linear outputs for Q values
    z3 = [0.0] * 4
    a3 = [0.0] * 4
    for j in range(4):
        total = biases[3][j]
        for i in range(16):
            total += a2[i] * weights[2][i][j]   # <- use a2
        z3[j] = total
        a3[j] = total   # linear output (raw Q values)

    return z1, a1, z2, a2, z3, a3

GAMMA = 0.9  # discount factor

def relu(x):
    return max(0, x)

def relu_derivative(x):
    return 1 if x > 0 else 0

def choose_action(output, epsilon=0.1):
    """ε-greedy policy for exploration."""
    if random.random() < epsilon:
        return random.randint(0, len(output) - 1)
    return output.index(max(output))

def backpropagate(inputLayer, target, learning_rate=0.01):
    """Backprop for a single example (MSE with linear output). Updates global weights/biases."""
    global weights, biases

    # Forward pass (store z and a)
    z1, a1, z2, a2, z3, a3 = feed_forward(inputLayer)

    # Output delta: dL/dz3 = (a3 - target) * 1  (linear output)
    delta3 = [0.0] * 4
    for j in range(4):
        # MSE derivative: d/dy (0.5*(y - t)^2) = (y - t)
        delta3[j] = (a3[j] - target[j])

    # Hidden layer 2 delta: delta2 = (W2 * delta3) * relu'(z2)
    delta2 = [0.0] * 16
    for i in range(16):
        downstream = 0.0
        for j in range(4):
            downstream += weights[2][i][j] * delta3[j]
        delta2[i] = downstream * relu_derivative(z2[i])

    # Hidden layer 1 delta: delta1 = (W1 * delta2) * relu'(z1)
    delta1 = [0.0] * 24
    for i in range(24):
        downstream = 0.0
        for j in range(16):
            downstream += weights[1][i][j] * delta2[j]
        delta1[i] = downstream * relu_derivative(z1[i])

    # Update weights and biases (gradient descent: w -= lr * grad)
    # weights[2] (16 -> 4)
    for i in range(16):
        for j in range(4):
            grad = delta3[j] * a2[i]   # dL/dw = delta_out_j * activation_prev_i
            weights[2][i][j] -= learning_rate * grad
    for j in range(4):
        biases[3][j] -= learning_rate * delta3[j]

    # weights[1] (24 -> 16)
    for i in range(24):
        for j in range(16):
            grad = delta2[j] * a1[i]
            weights[1][i][j] -= learning_rate * grad
    for j in range(16):
        biases[2][j] -= learning_rate * delta2[j]

    # weights[0] (input -> 24)
    for i in range(len(inputLayer)):
        for j in range(24):
            grad = delta1[j] * inputLayer[i]
            weights[0][i][j] -= learning_rate * grad
    for j in range(24):
        biases[1][j] -= learning_rate * delta1[j]

    return a3  # return Q-values if caller needs them

def train_Q_network(state, next_state, action, reward, done, lr=0.01):
    """Single Q-learning update using bootstrapped target (Q-learning)."""
    # get current Q and next Q
    _, _, _, _, _, q_values = feed_forward(state)
    _, _, _, _, _, next_q_values = feed_forward(next_state)

    target_q = q_values[:]   # copy
    if done:
        target = reward
    else:
        target = reward + GAMMA * max(next_q_values)
    target_q[action] = target

    # backprop using the state and the desired target_q
    backpropagate(state, target_q, learning_rate=lr)
